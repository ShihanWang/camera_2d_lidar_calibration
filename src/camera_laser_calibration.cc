
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <eigen3/Eigen/Core>
#include <Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace Eigen;
using namespace ceres;
using namespace cv;

typedef vector<Eigen::Vector3d> Vector3dPoints;
typedef vector<Eigen::Vector2d> Vector2dPoints;

double fx, fy, cx, cy;
cv::Mat K, D;


class ErrorTypes
{
public:
    ErrorTypes(Eigen::Vector2d observation) : observation_(observation) {}

    template <typename T>
    bool operator()(const T *const quaternion,
                    const T *const translation,
                    const T *const point,
                    T *residuals) const
    {

        Eigen::Map<const Eigen::Quaternion<T>> q_cl(quaternion);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_cl(translation);

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_l(point);
        Eigen::Matrix<T, 3, 1> p_c = q_cl * p_l + t_cl;

        residuals[0] = T(fx * p_c(0, 0) / p_c(2, 0) + cx - T(observation_(0)));
        residuals[1] = T(fy * p_c(1, 0) / p_c(2, 0) + cy - T(observation_(1)));

        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector2d observation)
    {
        return (new ceres::AutoDiffCostFunction<ErrorTypes, 2, 4, 3, 3>(
            new ErrorTypes(observation)));
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    Eigen::Vector2d observation_;
};

bool LoadData(const string &data_path, Vector3dPoints &laser_points, Vector2dPoints &image_points);
void BuildOptimizationProblem(Vector3dPoints &laser_points,
                              const Vector2dPoints &image_points,
                              Quaterniond &q,
                              Vector3d &t,
                              ceres::Problem *problem);
bool SolveOptimizationProblem(ceres::Problem *problem, bool show_the_solver_details = false);

void SetCameraParams(const string &file_name);

vector<Point3d> ConvertPointsFormat(const Vector3dPoints &points);

int main(int argc, char *argv[])
{
    if (argv[1] == nullptr)
    {
        cout << "Please Check the launch file" << endl;
    }
    string config_path(argv[1]);
    string data_path(argv[2]);
    string output_path(argv[3]);
    cout << "Config.path = " << config_path << endl;
    SetCameraParams(config_path);

    /// Put your initial guess here. Tcl which take a vector from laser to camera.
    Matrix3d init_R = Matrix3d::Identity();
    Quaterniond init_q(init_R);
    // Vector3d init_t(0.160323, -0.13979, -0.140524);
    Vector3d init_t(0.1, 0.5, 0.1);
    ceres::Problem problem;
    Vector3dPoints laser_points;
    Vector2dPoints image_points;
    cout << "data.path = " << data_path << endl;
    /// format: x y z w tx ty tz
    ofstream outFile(output_path);

    /**
     * Load data
     */
    cout << "Before Optimization\n"
         << "R = \n"
         << init_q.matrix() << "\nt = \n"
         << init_t.transpose() << endl;
    if (LoadData(data_path, laser_points, image_points))
    {
        cout << "Load data suscessfully!" << endl;

        /**
         * Optimizing
         */
        BuildOptimizationProblem(laser_points, image_points, init_q, init_t, &problem);
        SolveOptimizationProblem(&problem, true);

        cout << "After Optimization:\n";
        cout << "R = \n"
             << init_q.matrix() << endl;
        cout << "t = \n"
             << init_t.transpose() << endl;
        outFile << init_q.x() << " " << init_q.y() << " " << init_q.z() << " " << init_q.w()
                << " " << init_t(0) << " " << init_t(1) << " " << init_t(2) << endl;
        cout << init_q.x() << " " << init_q.y() << " " << init_q.z() << " " << init_q.w()
                << " " << init_t(0) << " " << init_t(1) << " " << init_t(2) << endl;        
        cout << "Result output format: qx qy qz qw tx ty tz" << endl;        
        vector<Point2d> projected_points;
        cv::Mat r_mat, r_vec;
        eigen2cv(init_q.matrix(), r_mat);
        Rodrigues(r_mat, r_vec);
        cv::Mat t_vec = (Mat_<double>(3, 1) << init_t(0), init_t(1), init_t(2));
        cv::projectPoints(ConvertPointsFormat(laser_points), r_vec, t_vec, K, D, projected_points);
        double sum = 0;
        double max_pixel = 0, min_pixel = 100, mean_pixel, sum_pixel = 0;        
        for(int i = 0; i < projected_points.size(); ++i)
        {
            double p = sqrt(pow(projected_points[i].x - image_points[i].x(),2) + pow(projected_points[i].y - image_points[i].y(),2));
            if(max_pixel < p)
            {
                max_pixel = p;
            }
            if(min_pixel > p)
            {
                min_pixel = p;
            }
            sum_pixel += p;
            sum += pow(projected_points[i].x - image_points[i].x(),2) + pow(projected_points[i].y - image_points[i].y(),2);
        }
        sum = sqrt(sum / projected_points.size());
        sum_pixel /= projected_points.size();
        cout << "Max in pixel =: " << max_pixel << endl;
        cout << "Min in pixel =: " << min_pixel << endl;
        cout << "Mean in pixel =: " << sum_pixel << endl;
        cout << "RMSE in pixel =: " << sum << endl;
        cout << "Save optimized result in: " << output_path << endl;
    }

    return 0;
}

bool LoadData(const string &data_path, Vector3dPoints &laser_points, Vector2dPoints &image_points)
{
    if (data_path.empty())
        return false;
    ifstream in_file;
    in_file.open(data_path);
    while (in_file.good())
    {
        Vector3d laser_point;
        Vector2d image_point;
        in_file >> laser_point(0) >> laser_point(1) >> image_point(0) >> image_point(1);
        laser_point(2) = 0.;
        laser_points.push_back(laser_point);
        image_points.push_back(image_point);
        in_file.get();
    }
    in_file.close();
    return true;
}

void BuildOptimizationProblem(Vector3dPoints &laser_points,
                              const Vector2dPoints &image_points,
                              Quaterniond &q,
                              Vector3d &t,
                              ceres::Problem *problem)
{
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    ceres::LocalParameterization *quaternion_local_parameterization =
        new EigenQuaternionParameterization;

    for (int i = 0; i < laser_points.size(); ++i)
    {
        ceres::CostFunction *cost_function =
            ErrorTypes::Create(image_points[i]);
        problem->AddResidualBlock(cost_function,
                                  loss_function,
                                  q.coeffs().data(),
                                  t.data(),
                                  laser_points[i].data());
        problem->SetParameterization(q.coeffs().data(),
                                     quaternion_local_parameterization);
    }

    for (int j = 0; j < laser_points.size(); ++j)
    {
        problem->SetParameterBlockConstant(laser_points[j].data());
    }
}

bool SolveOptimizationProblem(ceres::Problem *problem, bool show_the_solver_details)
{

    ceres::Solver::Options options;
    options.max_num_iterations = 1000;
    options.linear_solver_type = ceres::DENSE_QR;

    ceres::Solver::Summary summary;
    ceres::Solve(options, problem, &summary);
    if (show_the_solver_details)
        std::cout << summary.FullReport() << '\n';

    return summary.IsSolutionUsable();
}

void SetCameraParams(const string &file_name)
{
    cv::FileStorage file = cv::FileStorage(file_name, cv::FileStorage::READ);
    if (!file.isOpened())
    {
        cerr << "parameter file " << file_name << " dose not exist.." << endl;
        exit(-1);
    }
    fx = file["fx"];
    fy = file["fy"];
    cx = file["cx"];
    cy = file["cy"];
    K = (Mat_<double>(3, 3) << fx, 0, cx,
                              0, fy, cy,
                              0, 0, 1);
    D = (Mat_<double>(5, 1) << file["k1"], file["k2"], file["p1/k3"], file["p2/k4"], 0);
    cout << "K : " << K << endl;
    cout << "D : " << D << endl;
    file.release();
}

vector<Point3d> ConvertPointsFormat(const Vector3dPoints &points)
{
    vector<Point3d> pts;
    if (points.size() == 0)
        return pts;
    for (auto pt : points)
    {
        pts.push_back(Point3d(pt.x(), pt.y(), pt.z()));
    }
    return pts;
}