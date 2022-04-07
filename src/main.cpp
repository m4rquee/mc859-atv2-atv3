/* Copyright 2022, Gurobi Optimization, LLC */

/* Solve a traveling salesman problem on a randomly generated set of
   points using lazy constraints. The base MIP model only includes
   'degree-2' constraints, requiring each node to have exactly
   two incident edges. Solutions to this model may contain subtours -
   tours that don't visit every node. The lazy constraint callback
   adds new constraints to cut them off. */

#include "gurobi_c++.h"
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <sstream>

using namespace std;

string itos(int i) {
  stringstream s;
  s << i;
  return s.str();
}

// Euclidean distance between points 'i' and 'j'.
inline double distance(const double *x, const double *y, int i, int j) {
  double dx = x[i] - x[j];
  double dy = y[i] - y[j];
  return sqrt(dx * dx + dy * dy);
}

// Given an integer-feasible solution 'sol', find the smallest
// sub-tour. Result is returned in 'tour', and length is
// returned in 'tourlenP'.
void findsubtour(int n, double **sol, int *tourlenP, int *tour) {
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
  *tourlenP = bestlen;

  delete[] seen;
}

// Subtour elimination callback.  Whenever a feasible solution is found,
// find the smallest subtour, and add a subtour elimination constraint
// if the tour doesn't visit every node.
class SubTourElim : public GRBCallback {
public:
  int n;
  GRBVar **vars;

  SubTourElim(GRBVar **xvars, int xn) : n(xn), vars(xvars) {}

protected:
  void callback() override {
    try {
      if (where == GRB_CB_MIPSOL) {
        // Found an integer feasible solution - does it visit every node?
        auto x = new double *[n];
        int *tour = new int[n];
        for (int i = 0; i < n; i++)
          x[i] = getSolution(vars[i], n);

        int len;
        findsubtour(n, x, &len, tour);

        if (len < n) {
          // Add subtour elimination constraint:
          GRBLinExpr expr = 0;
          for (int i = 0; i < len; i++)
            for (int j = i + 1; j < len; j++)
              expr += vars[tour[i]][tour[j]];
          addLazy(expr <= len - 1);
        }

        for (int i = 0; i < n; i++)
          delete[] x[i];
        delete[] x;
        delete[] tour;
      }
    } catch (GRBException &e) {
      cout << "Error number: " << e.getErrorCode() << endl;
      cout << e.getMessage() << endl;
    } catch (...) {
      cout << "Error during callback" << endl;
    }
  }
};

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cout << "Usage: tsp_c++ size" << endl;
    return 1;
  }

  int n = atoi(argv[1]);
  auto x = new double[n];
  auto y = new double[n];

  for (int i = 0; i < n; i++) {
    x[i] = ((double)rand()) / RAND_MAX;
    y[i] = ((double)rand()) / RAND_MAX;
  }

  GRBEnv *env = nullptr;
  GRBVar **vars = nullptr;

  vars = new GRBVar *[n];
  for (int i = 0; i < n; i++)
    vars[i] = new GRBVar[n];

  try {
    env = new GRBEnv();
    GRBModel model = GRBModel(*env);

    // Must set LazyConstraints parameter when using lazy constraints:
    model.set(GRB_IntParam_LazyConstraints, 1);

    // Create binary decision variables:
    for (int i = 0; i < n; i++)
      for (int j = 0; j <= i; j++) {
        vars[i][j] = model.addVar(0.0, 1.0, distance(x, y, i, j), GRB_BINARY,
                                  "x_" + itos(i) + "_" + itos(j));
        vars[j][i] = vars[i][j];
      }

    // Degree-2 constraints:
    for (int i = 0; i < n; i++) {
      GRBLinExpr expr = 0;
      for (int j = 0; j < n; j++)
        expr += vars[i][j];
      model.addConstr(expr == 2, "deg2_" + itos(i));
    }

    // Forbid edge from node back to itself:
    for (int i = 0; i < n; i++)
      vars[i][i].set(GRB_DoubleAttr_UB, 0);

    // Set callback function:
    SubTourElim cb = SubTourElim(vars, n);
    model.setCallback(&cb);

    // Optimize model:
    model.optimize();

    // Extract solution:
    if (model.get(GRB_IntAttr_SolCount) > 0) {
      auto sol = new double *[n];
      for (int i = 0; i < n; i++)
        sol[i] = model.get(GRB_DoubleAttr_X, vars[i], n);

      int *tour = new int[n];
      int len;

      findsubtour(n, sol, &len, tour);
      assert(len == n);

      cout << "Tour: ";
      for (int i = 0; i < len; i++)
        cout << tour[i] << " ";
      cout << endl;

      for (int i = 0; i < n; i++)
        delete[] sol[i];
      delete[] sol;
      delete[] tour;
    }
  } catch (GRBException &e) {
    cout << "Error number: " << e.getErrorCode() << endl;
    cout << e.getMessage() << endl;
  } catch (...) {
    cout << "Error during optimization" << endl;
  }

  for (int i = 0; i < n; i++)
    delete[] vars[i];
  delete[] vars;
  delete[] x;
  delete[] y;
  delete env;
  return 0;
}
