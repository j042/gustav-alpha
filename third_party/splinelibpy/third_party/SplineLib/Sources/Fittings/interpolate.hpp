#include "Sources/Fittings/utils.hpp"
#include "Sources/Fittings/lu_solve.hpp"

void InterpolateCurve(double* points,
                      int& num_points,
                      int& dim,
                      int& degree,
                      bool centripetal,
                      std::vector<double>& knot_vector,
                      std::vector<double>& control_points);

void InterpolateSurface(double* points,
                        int& num_points,
                        int& dim,
                        int& degree_u,
                        int& degree_v,
                        int& size_u,
                        int& size_v,
                        bool centripetal,
                        std::vector<double>& knot_vector_u,
                        std::vector<double>& knot_vector_v,
                        std::vector<double>& control_points);
