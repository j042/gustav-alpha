#include "Sources/Fittings/interpolate.hpp"

void InterpolateCurve(double* points,
                      int& num_points,
                      int& dim,
                      int& degree,
                      bool centripetal,
                      std::vector<double>& knot_vector,
                      std::vector<double>& control_points) {

    std::vector<double> u_k, coefficient_matrix;


    u_k = ParametrizeCurve(points, num_points, dim, centripetal);

    knot_vector = ComputeKnotVector(degree, num_points, u_k);

    coefficient_matrix = BuildCoefficientMatrix(degree, knot_vector, u_k, points, num_points);

    control_points = LUSolve(coefficient_matrix, points, num_points, dim);
}

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
                        std::vector<double>& control_points) {

    std::vector<double> u_k, v_l, coefficient_matrix, tmp_result, tmp_control_points{};
    int u, v, i;
    double* pts_u = new double[size_u * dim];
    double* pts_v = new double[size_v * dim];

    // ParametrizeSurface
    ParametrizeSurface(points, num_points, dim, size_u, size_v, centripetal, u_k, v_l);

    knot_vector_u = ComputeKnotVector(degree_u, size_u, u_k);

    knot_vector_v = ComputeKnotVector(degree_v, size_v, v_l);

    // u - direction global interpolation
    for (v = 0; v < size_v; v++) {
        for (u = 0; u < size_u; u++) {
            for (i = 0; i < dim; i++) {
                pts_u[u * dim + i] = points[(v + (size_v * u)) * dim + i];
            }
        }
        coefficient_matrix = BuildCoefficientMatrix(degree_u, knot_vector_u, u_k, pts_u, size_u);
        tmp_result = LUSolve(coefficient_matrix, pts_u, size_u, dim); 
        std::move(tmp_result.begin(), tmp_result.end(), std::back_inserter(tmp_control_points));
    }

    // v - direction global interpolation
    control_points.clear();
    for (u = 0; u < size_u; u++) {
        for (v = 0; v < size_v; v++) {
            for (i = 0; i < dim; i++) {
                pts_v[v * dim + i] = tmp_control_points[(u + (size_u * v)) * dim + i];
            }
        }
        coefficient_matrix = BuildCoefficientMatrix(degree_v, knot_vector_v, v_l, pts_v, size_v);
        tmp_result = LUSolve(coefficient_matrix, pts_v, size_v, dim); 
        std::move(tmp_result.begin(), tmp_result.end(), std::back_inserter(control_points));
    }
    
    delete[] pts_u, pts_v;

}
