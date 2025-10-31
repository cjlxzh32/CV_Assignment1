#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include "json.hpp"
using json = nlohmann::json;


static Eigen::Vector3d RotMatToAngleAxis(const Eigen::Matrix3d& R) {
    double R_arr[9] = {
        R(0,0), R(0,1), R(0,2),
        R(1,0), R(1,1), R(1,2),
        R(2,0), R(2,1), R(2,2)
    };
    double aa[3];
    ceres::RotationMatrixToAngleAxis(R_arr, aa);
    return Eigen::Vector3d(aa[0], aa[1], aa[2]);
}


static Eigen::Matrix3d AngleAxisToRotMat(const Eigen::Vector3d& aa) {
    double R_arr[9];
    double aa_arr[3] = {aa[0], aa[1], aa[2]};
    ceres::AngleAxisToRotationMatrix(aa_arr, R_arr);
    Eigen::Matrix3d R;
    R << R_arr[0], R_arr[1], R_arr[2],
         R_arr[3], R_arr[4], R_arr[5],
         R_arr[6], R_arr[7], R_arr[8];
    return R;
}


struct ReprojError {
    ReprojError(double u, double v,
                double fx, double fy,
                double cx, double cy)
        : u_(u), v_(v),
          fx_(fx), fy_(fy),
          cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const cam,
                    const T* const point,
                    T* residuals) const {

        const T* angle_axis = cam + 0;
        const T* trans      = cam + 3;

        T pc[3];
        ceres::AngleAxisRotatePoint(angle_axis, point, pc);
        pc[0] += trans[0];
        pc[1] += trans[1];
        pc[2] += trans[2];

        T xp = pc[0] / pc[2];
        T yp = pc[1] / pc[2];

        T u_pred = T(fx_) * xp + T(cx_);
        T v_pred = T(fy_) * yp + T(cy_);

        residuals[0] = u_pred - T(u_);
        residuals[1] = v_pred - T(v_);
        return true;
    }

    static ceres::CostFunction* Create(double u, double v,
                                       double fx, double fy,
                                       double cx, double cy) {
        return (new ceres::AutoDiffCostFunction<
            ReprojError, 2 ,
                         6 ,
                         3 >(
            new ReprojError(u, v, fx, fy, cx, cy)));
    }

    double u_, v_;
    double fx_, fy_, cx_, cy_;
};

int main(int argc, char** argv) {

    std::string input_path  = "result/arch/ba_problem_export.json";
    std::string output_path = "result/arch/ba_problem_ceres_refined.json";
    bool fix_first_camera   = true;
    double huber_delta      = 3.0;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--input" && i + 1 < argc) {
            input_path = argv[++i];
        } else if (k == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (k == "--fix_first_camera" && i + 1 < argc) {
            fix_first_camera = (std::stoi(argv[++i]) != 0);
        } else if (k == "--huber_delta" && i + 1 < argc) {
            huber_delta = std::stod(argv[++i]);
        }
    }

    std::cout << "[INFO] input_path  = " << input_path  << "\n";
    std::cout << "[INFO] output_path = " << output_path << "\n";
    std::cout << "[INFO] fix_first_camera = " << (fix_first_camera?1:0) << "\n";
    std::cout << "[INFO] huber_delta = " << huber_delta << "\n";


    std::ifstream ifs(input_path);
    if (!ifs.is_open()) {
        std::cerr << "[ERR] cannot open " << input_path << "\n";
        return 1;
    }
    json j;
    ifs >> j;


    auto K_json = j["K"];
    Eigen::Matrix3d K;
    K << K_json[0][0], K_json[0][1], K_json[0][2],
         K_json[1][0], K_json[1][1], K_json[1][2],
         K_json[2][0], K_json[2][1], K_json[2][2];

    double fx = K(0,0);
    double fy = K(1,1);
    double cx = K(0,2);
    double cy = K(1,2);


    auto cams_json = j["cameras"];
    auto pts_json  = j["points"];
    auto obs_json  = j["observations"];

    const int num_cams = cams_json.size();
    const int num_pts  = pts_json.size();
    const int num_obs  = obs_json.size();

    std::cout << "[INFO] cams=" << num_cams
              << " pts=" << num_pts
              << " obs=" << num_obs << "\n";


    std::vector<Eigen::VectorXd> cam_params(num_cams);
    for (int i = 0; i < num_cams; ++i) {
        cam_params[i].resize(6);
    }


    std::vector<Eigen::Vector3d> pt_params(num_pts);


    for (auto& c : cams_json) {
        int cid = c["id"];

        auto Rj = c["R_w2c"];
        Eigen::Matrix3d R_w2c;
        R_w2c << Rj[0][0], Rj[0][1], Rj[0][2],
                 Rj[1][0], Rj[1][1], Rj[1][2],
                 Rj[2][0], Rj[2][1], Rj[2][2];

        auto tj = c["t_w2c"];
        Eigen::Vector3d t_w2c(tj[0], tj[1], tj[2]);

        Eigen::Vector3d aa = RotMatToAngleAxis(R_w2c);

        cam_params[cid].segment<3>(0) = aa;
        cam_params[cid].segment<3>(3) = t_w2c;
    }


    for (auto& p : pts_json) {
        int pid = p["id"];
        auto Xj = p["X"];
        pt_params[pid] = Eigen::Vector3d(Xj[0], Xj[1], Xj[2]);
    }


    ceres::Problem problem;

    for (auto& o : obs_json) {
        int cam_id = o["cam_id"];
        int pt_id  = o["pt_id"];
        auto uv    = o["uv"];

        double u = uv[0];
        double v = uv[1];

        ceres::CostFunction* cost =
            ReprojError::Create(u, v, fx, fy, cx, cy);

        ceres::LossFunction* loss =
            new ceres::HuberLoss(huber_delta);

        problem.AddResidualBlock(
            cost,
            loss,
            cam_params[cam_id].data(),
            pt_params[pt_id].data()
        );
    }


    if (fix_first_camera && num_cams > 0) {
        problem.SetParameterBlockConstant(cam_params[0].data());
        std::cout << "[INFO] camera 0 is fixed (gauge)\n";
    }


    ceres::Solver::Options options;

    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 50;
    options.num_threads = 8;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    std::cout << "===== CERES SUMMARY =====\n";
    std::cout << summary.BriefReport() << "\n";


    json out;
    out["K"] = j["K"];

    {
        json cams_out = json::array();
        for (int cid = 0; cid < num_cams; ++cid) {
            Eigen::Vector3d aa = cam_params[cid].segment<3>(0);
            Eigen::Vector3d t  = cam_params[cid].segment<3>(3);

            Eigen::Matrix3d R_w2c = AngleAxisToRotMat(aa);

            json one_cam;
            one_cam["id"] = cid;

            one_cam["R_w2c"] = {
                { R_w2c(0,0), R_w2c(0,1), R_w2c(0,2) },
                { R_w2c(1,0), R_w2c(1,1), R_w2c(1,2) },
                { R_w2c(2,0), R_w2c(2,1), R_w2c(2,2) }
            };

            one_cam["t_w2c"] = { t[0], t[1], t[2] };

            cams_out.push_back(one_cam);
        }
        out["cameras_optimized"] = cams_out;
    }

    {
        json pts_out = json::array();
        for (int pid = 0; pid < num_pts; ++pid) {
            Eigen::Vector3d X = pt_params[pid];
            json one_pt;
            one_pt["id"] = pid;
            one_pt["X"]  = { X[0], X[1], X[2] };
            pts_out.push_back(one_pt);
        }
        out["points_optimized"] = pts_out;
    }

    out["notes"] = {
        { "solver", "Ceres Solver BA (sparse Schur)" },
        { "fix_first_camera", fix_first_camera },
        { "huber_delta_px", huber_delta },
        { "num_cameras", num_cams },
        { "num_points", num_pts },
        { "num_observations", num_obs },
        { "convention", "R_w2c, t_w2c: x_cam = R_w2c * x_world + t_w2c" }
    };

    std::ofstream ofs(output_path);
    if (!ofs.is_open()) {
        std::cerr << "[ERR] cannot write " << output_path << "\n";
        return 1;
    }
    ofs << out.dump(2);
    ofs.close();

    std::cout << "[OK] wrote refined BA result to " << output_path << "\n";
    return 0;
}
