/* Solve a k-similar traveling salesman problem on a set of
   points using lazy constraints. The base MIP model only includes
   'degree-2' and similarity constraints, requiring each node to have exactly
   two incident edges and each tour to share at least k edges. Solutions to this
   model may contain subtours - tours that don't visit every node. The lazy
   constraint callback adds new constraints to cut them off. */

#include "gurobi_c++.h"
#include <cmath>
#include <cstdlib>
#include <sstream>

using namespace std;

string itos(int i) {
  stringstream s;
  s << i;
  return s.str();
}

// Ceil of the euclidean distance between points 'i' and 'j'.
inline double distance(const double *x, const double *y, int i, int j) {
  double dx = x[i] - x[j];
  double dy = y[i] - y[j];
  return ceil(sqrt(dx * dx + dy * dy));
}

inline int mod(int a, int b) {
  if (b < 0)
    return -mod(-a, -b);
  int ret = a % b;
  if (ret < 0)
    ret += b;
  return ret;
}

// Given an integer-feasible solution 'sol', find the smallest
// sub-tour. Result is returned in 'tour', and length is
// returned in 'tour_len'.
void find_subtour(int n, double **sol, int *tour_len, int *tour) {
  bool *seen = new bool[n];
  int bestind, bestlen;
  int i, node, len, start;

  for (i = 0; i < n; i++)
    seen[i] = false;

  start = 0;
  bestlen = n + 1;
  bestind = -1;
  while (start < n) {
    for (node = 0; node < n; node++)
      if (!seen[node])
        break;
    if (node == n)
      break;
    for (len = 0; len < n; len++) {
      tour[start + len] = node;
      seen[node] = true;
      for (i = 0; i < n; i++)
        if (sol[node][i] > 0.5 && !seen[i]) {
          node = i;
          break;
        }
      if (i == n) {
        len++;
        if (len < bestlen) {
          bestlen = len;
          bestind = start;
        }
        start += len;
        break;
      }
    }
  }

  for (i = 0; i < bestlen; i++)
    tour[i] = tour[bestind + i];
  *tour_len = bestlen;

  delete[] seen;
}

// Subtour elimination callback. Whenever a feasible solution is found,
// find the smallest subtour, and add a subtour elimination constraint
// if the tour doesn't visit every node.
class SubTourElim : public GRBCallback {
public:
  int n, k;
  GRBVar **X1, **X2;
  double **x1, **x2;
  int *tour1, *tour2;

  SubTourElim(GRBVar **xX1, GRBVar **xX2, int xn, int xk)
      : n(xn), k(xk), X1(xX1), X2(xX2) {
    x1 = new double *[n], x2 = new double *[n];
    tour1 = new int[n], tour2 = new int[n];
  }

protected:
  bool eliminate_min_sub_tour(GRBVar **vars, double **x, int *tour) {
    for (int i = 0; i < n; i++)
      x[i] = getSolution(vars[i], n);

    int len;
    find_subtour(n, x, &len, tour);

    bool eliminated = false;
    if (len < n) {
      // Add subtour elimination constraint:
      GRBLinExpr expr = 0;
      for (int i = 0; i < len; i++)
        for (int j = i + 1; j < len; j++)
          expr += vars[tour[i]][tour[j]];
      addLazy(expr <= len - 1);
      eliminated = true;
    }

    return eliminated;
  }

  int similarity() {
    int num_shared = 0;
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++)
        num_shared += x1[i][j] > 0.5 && x2[i][j] > 0.5;
    return num_shared;
  }

  void print_edges(double **x) const {
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++)
        cout << (x[i][j] > 0.5);
      cout << "\n";
    }
    cout << "\n";
  }

  void callback() override {
    try {
      if (where == GRB_CB_MIPSOL) {
        // Found an integer feasible solution - does it visit every node?
        bool eliminated = eliminate_min_sub_tour(X1, x1, tour1);
        eliminated |= eliminate_min_sub_tour(X2, x2, tour2);
#ifdef DEBUG_MODE
        print_edges(x1);
        print_edges(x2);
#endif

        if (!eliminated) { // found an upper bound to the IP model
          // Fix the similarity violation and generates a new valid solution
          // to the non-relaxed IP model:
          int missing_k = k - similarity();
          while (missing_k > 0)
            lagrangian_heuristic(missing_k);
        }

        // Cleanup the results:
        for (int i = 0; i < n; i++)
          delete[] x1[i];
        for (int i = 0; i < n; i++)
          delete[] x2[i];
      }
    } catch (GRBException &e) {
      cout << "Error number: " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
    } catch (...) {
      cout << "Error during callback" << endl;
    }
  }

  inline int similarity_delta(int u, int v, int s, int t) const {
    return 1 + (x1[s][t] > 0.5) - (x1[u][s] > 0.5) - (x1[v][t] > 0.5);
  }

  void flip(int u, int v, int s, int t) const {
    // Flip the edges:
    x2[u][v] = x2[v][u] = x2[s][t] = x2[t][s] = 1;
    x2[u][s] = x2[s][u] = x2[v][t] = x2[t][v] = 0;
    // Reorder a subsequence of the tour to make the flip valid:
    bool *seen = new bool[n];
    seen[0] = true;
    for (int i = 1; i < n; i++)
      seen[i] = false;
    tour2[0] = 0;
    for (int i = 0; i < n - 1; i++)
      for (int j = 0; j < n; j++)
        if (x2[tour2[i]][j] > 0.5 && !seen[j]) { // the next node is j
          tour2[i + 1] = j;
          seen[j] = true;
          break;
        }
    delete[] seen;
  }

  inline void get_endpoint_neighbors(int u, int v, int &u_l, int &u_r, int &v_l,
                                     int &v_r) const {
    for (int i = 0; i < n; i++)
      if (tour2[i] == u) // found an end point of uv
        u_l = tour2[mod(i - 1, n)], u_r = tour2[mod(i + 1, n)];
      else if (tour2[i] == v)
        v_l = tour2[mod(i + 1, n)], v_r = tour2[mod(i - 1, n)];
  }

  void lagrangian_heuristic(int &missing_k) {
    int u, v, s, t;
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++)
        // ij is a candidate to be added to tour 2:
        if (x1[i][j] > 0.5 && x2[i][j] <= 0.5) {
          int delta, i_l, i_r, j_l, j_r;
          get_endpoint_neighbors(i, j, i_l, i_r, j_l, j_r);
          if ((delta = similarity_delta(i, j, i_l, j_r)) > 0) {
            flip(i, j, i_l, j_r);
            missing_k -= delta; // increased the similarity by an edge pair flip
            return;
          } else if (delta >= 0)
            u = i, v = j, s = i_l,
            t = j_r; // found at least a similarity keeping edge pair

          if ((delta = similarity_delta(i, j, i_r, j_l)) > 0) {
            flip(i, j, i_r, j_l);
            missing_k -= delta; // increased the similarity by an edge pair flip
            return;
          } else if (delta >= 0)
            u = i, v = j, s = i_r,
            t = j_l; // found at least a similarity keeping edge pair
        }

    // Will make a similarity increasing pair appear:
    flip(u, v, s, t);
  }
};

void read_data(double *x1, double *y1, double *x2, double *y2, int n) {
  for (int i = 0; i < n; i++)
    cin >> x1[i] >> y1[i] >> x2[i] >> y2[i];
}

void subgradient_method(int n, int k) {
  // Read the points coordinates data: -----------------------------------------
  auto x1 = new double[n], y1 = new double[n];
  auto x2 = new double[n], y2 = new double[n];
  read_data(x1, y1, x2, y2, n);

  // The similarity constraint will be dualized: -------------------------------
  int n_duals = 1;
  auto lambda = new double[n_duals], g_k = new double[n_duals];
  double pi_k = 2.0, g_k_srq_sum = 0, Z_LB_k, Z_UB, eps = 1E-2;

  // Initialize the Lagrangian multipliers lambda:
  for (int i = 0; i < n_duals; i++)
    lambda[i] = 0;

  // Costs of each edge for each salesman and if an edge is required to be
  // doubled: ------------------------------------------------------------------
  auto X1 = new GRBVar *[n], X2 = new GRBVar *[n], d = new GRBVar *[n];
  for (int i = 0; i < n; i++) {
    X1[i] = new GRBVar[n];
    X2[i] = new GRBVar[n];
    d[i] = new GRBVar[n];
  }

  double remaining_time = 1800;
  try {
    // Create the gurobi model used to solve the LLBP relaxations: -------------
    auto *env = new GRBEnv();
    env->set(GRB_DoubleParam_TimeLimit, remaining_time);

    GRBModel model = GRBModel(*env);
    model.set(GRB_StringAttr_ModelName, "Atividade 3");
    model.set(GRB_IntAttr_ModelSense, GRB_MINIMIZE);

    // Must set LazyConstraints parameter when using lazy constraints:
    model.set(GRB_IntParam_LazyConstraints, 1);

    // Focus primarily on feasibility of the relaxation:
    model.set(GRB_IntParam_MIPFocus, GRB_MIPFOCUS_FEASIBILITY);
    model.set(GRB_IntParam_Cuts, GRB_CUTS_AGGRESSIVE);
    model.set(GRB_IntParam_Presolve, GRB_PRESOLVE_AGGRESSIVE);

    // Create binary decision variables:
    for (int i = 0; i < n; i++)
      for (int j = 0; j <= i; j++) {
        X1[i][j] = model.addVar(0.0, 1.0, distance(x1, y1, i, j), GRB_BINARY,
                                "x1_" + itos(i) + "_" + itos(j));
        X1[j][i] = X1[i][j];
        X2[i][j] = model.addVar(0.0, 1.0, distance(x2, y2, i, j), GRB_BINARY,
                                "x2_" + itos(i) + "_" + itos(j));
        X2[j][i] = X2[i][j];
        d[i][j] = model.addVar(0.0, 1.0, 0, GRB_BINARY,
                               "d_" + itos(i) + "_" + itos(j));
        d[j][i] = d[i][j];
      }
    model.update(); // run update to use model inserted variables

    // Similarity constraint between tours (1st constraint):
    GRBLinExpr expr = 0;
    for (int i = 0; i < n; i++)
      for (int j = i + 1; j < n; j++)
        expr += d[i][j];
    model.addConstr(expr == n, "similarity");

    // Degree-2 constraints:
    for (int i = 0; i < n; i++) {
      GRBLinExpr expr1 = 0, expr2 = 0;
      for (int j = 0; j < n; j++) {
        expr1 += X1[i][j];
        expr2 += X2[i][j];
      }
      model.addConstr(expr1 == 2, "deg2_1_" + itos(i));
      model.addConstr(expr2 == 2, "deg2_2_" + itos(i));
    }

    // Forbid edge from node back to itself:
    for (int i = 0; i < n; i++) {
      X1[i][i].set(GRB_DoubleAttr_UB, 0);
      X2[i][i].set(GRB_DoubleAttr_UB, 0);
      d[i][i].set(GRB_DoubleAttr_UB, 0);
    }

    // If d_ij then both edges should be used, otherwise no restrictions are
    // imposed:
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        model.addConstr(X1[i][j] >= d[i][j],
                        "double1_" + itos(i) + "_" + itos(j));
        model.addConstr(X2[i][j] >= d[i][j],
                        "double2_" + itos(i) + "_" + itos(j));
      }

    // Set callback function:
    SubTourElim cb = SubTourElim(X1, X2, n, k);
    model.setCallback(&cb);

    // Solve first with k = |V| = n to get an UB: ------------------------------
    model.optimize();
    Z_UB = model.getObjective().getValue();
    model.remove(model.getConstrs()[0]); // the 1st constraint (similarity)
    model.update();

    for (int iter = 0; iter < 1000; iter++) {
      // The -lambda[0] obj is because of the dualized similarity constraint:
      for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
          d[i][j].set(GRB_DoubleAttr_Obj, -lambda[0]);

      // Solve the LLBP and advance one step of the subgradients method: -------
      remaining_time -= model.get(GRB_DoubleAttr_Runtime);
      if (remaining_time <= 0)
        break;
      env->set(GRB_DoubleParam_TimeLimit, remaining_time);
      model.optimize(); // solve the LLBP
      // lambda[0] * k is a constant for fixed lambda:
      Z_LB_k = model.getObjective().getValue() + lambda[0] * k;

      // Compute the subgradients:
      for (int i = 0; i < n_duals; i++) {
        g_k[i] = k;
        for (int u = 0; u < n; u++)
          for (int v = u + 1; v < n; v++)
            g_k[i] -= d[u][v].get(GRB_DoubleAttr_X);
        g_k_srq_sum += g_k[i] * g_k[i];
      }

      // Update the lagrangian multipliers:
      auto alpha_k = pi_k * (Z_UB - Z_LB_k) / g_k_srq_sum;
      for (int i = 0; i < n_duals; i++)
        lambda[i] = std::max(0.0, lambda[i] + alpha_k * g_k[i]);

      pi_k *= 0.9; // decrease the pi value to guarantee convergence
      cout << "\niter = " << iter << "; lambda = " << lambda[0]
           << "; alpha = " << alpha_k << "; LB = " << Z_LB_k
           << "; UB = " << Z_UB << "\n\n";
      if (pi_k < eps)
        break;
    }
  } catch (GRBException &e) {
    cout << "Error number: " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch (...) {
    cout << "Error during callback" << endl;
  }
  for (int i = 0; i < n; i++) {
    delete[] X1[i];
    delete[] X2[i];
  }
  delete[] X1;
  delete[] X2;
  delete[] x1;
  delete[] x2;
  delete[] y1;
  delete[] y2;
  delete[] lambda;
  delete[] g_k;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    cout << "Usage: mc859 size k < coordinates_file" << endl;
    return 1;
  }
  subgradient_method(atoi(argv[1]), atoi(argv[2]));

  /*try {
    // Extract solution:
    if (model.get(GRB_IntAttr_SolCount) > 0) {
      auto sol1 = new double *[n], sol2 = new double *[n];
      for (int i = 0; i < n; i++) {
        sol1[i] = model.get(GRB_DoubleAttr_X, X1[i], n);
        sol2[i] = model.get(GRB_DoubleAttr_X, X2[i], n);
      }

      int *tour1 = new int[n], *tour2 = new int[n];
      int len1, len2;

      find_subtour(n, sol1, &len1, tour1);
      find_subtour(n, sol2, &len2, tour2);
      assert(len1 == n);
      assert(len2 == n);

      cout << "Tour 1: ";
      for (int i = 0; i < n; i++)
        cout << tour1[i] << " ";
      cout << endl;

      cout << "Tour 2: ";
      for (int i = 0; i < n; i++)
        cout << tour2[i] << " ";
      cout << endl;

      cout << "Shared edges:\n";
      for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
          if (d[i][j].get(GRB_DoubleAttr_X) > 0.5)
            cout << "(" << i << ", " << j << ") ";
      cout << endl;

      for (int i = 0; i < n; i++) {
        delete[] sol1[i];
        delete[] sol2[i];
      }
      delete[] sol1;
      delete[] sol2;
      delete[] tour1;
      delete[] tour2;
    }
  } catch (GRBException &e) {
    cout << "Error number: " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch (...) {
    cout << "Error during optimization" << endl;
  }*/

  return 0;
}
