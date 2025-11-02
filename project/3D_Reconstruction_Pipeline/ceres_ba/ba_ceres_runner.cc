// ba_ceres_runner.cc  (no SubsetParameterization; lock translation via bounds)
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <set>
#include "json.hpp"
using json = nlohmann::json;

// ============ 工具：旋转互转 ============
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

// ============ 方向模式 & 像素映射 ============
enum Mode { NONE, CW90, CCW90 };
static inline Mode ParseMode(const std::string& s) {
    if (s == "cw90")  return CW90;
    if (s == "ccw90") return CCW90;
    return NONE;
}
static inline void RemapObsToK(int W, int H, Mode m, double u, double v,
                               double& u_out, double& v_out) {
    if (m == NONE) { u_out = u; v_out = v; return; }
    if (m == CW90) { u_out = (double)(H - 1) - v; v_out = u; return; }
    if (m == CCW90) { u_out = v; v_out = (double)(W - 1) - u; return; }
}

// ============ 重投影残差 ============
struct ReprojError {
    ReprojError(double u, double v,
                double fx, double fy,
                double cx, double cy,
                double zmin)
        : u_(u), v_(v), fx_(fx), fy_(fy), cx_(cx), cy_(cy), zmin_(zmin) {}

    template <typename T>
    bool operator()(const T* const cam, const T* const point, T* residuals) const {
        const T* aa = cam;
        const T* t  = cam + 3;
        T pc[3];
        ceres::AngleAxisRotatePoint(aa, point, pc);
        pc[0] += t[0]; pc[1] += t[1]; pc[2] += t[2];

        if (pc[2] < T(zmin_)) { residuals[0] = T(0); residuals[1] = T(0); return true; }

        T xp = pc[0] / pc[2];
        T yp = pc[1] / pc[2];
        T u_pred = T(fx_) * xp + T(cx_);
        T v_pred = T(fy_) * yp + T(cy_);
        residuals[0] = T(u_) - u_pred;
        residuals[1] = T(v_) - v_pred;
        return true;
    }

    static ceres::CostFunction* Create(double u, double v,
                                       double fx, double fy,
                                       double cx, double cy,
                                       double zmin) {
        return new ceres::AutoDiffCostFunction<ReprojError, 2, 6, 3>(
            new ReprojError(u, v, fx, fy, cx, cy, zmin));
    }

    double u_, v_, fx_, fy_, cx_, cy_, zmin_;
};

// ============ JSON 工具 ============
static json load_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) { std::cerr << "[ERR] Cannot open " << path << std::endl; exit(1); }
    json j; f >> j; return j;
}
static void save_json(const std::string& path, const json& j) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) { std::cerr << "[ERR] Cannot write " << path << std::endl; exit(1); }
    ofs << j.dump(2); ofs.close();
}

// ============ 主程序 ============
int main(int argc, char** argv) {
    std::string input_path  = "result/arch/ba_problem_export.json";
    std::string output_path = "result/arch/ba_problem_ceres_refined.json";
    std::string cam_k_mode_map = ""; // JSON: { "0":"none","1":"cw90","2":"ccw90", ... }

    int  two_stage       = 1;
    int  fix_cam_prefix  = 1;   // 固定前 N 台相机
    int  stageB_free_t   = 0;   // Stage-B 是否放开平移
    int  min_track_len   = 2;
    int  img_w = 2160, img_h = 3840;

    double huber_delta      = 3.0;
    double prefilter_thresh = 0.0;
    double zmin             = 0.05;

    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i], v;
        auto next = [&]{ if (i + 1 < argc) v = argv[++i]; };
        if (k == "--input") { next(); input_path = v; }
        else if (k == "--output") { next(); output_path = v; }
        else if (k == "--two_stage") { next(); two_stage = std::stoi(v); }
        else if (k == "--fix_cam_prefix") { next(); fix_cam_prefix = std::stoi(v); }
        else if (k == "--stageB_free_t") { next(); stageB_free_t = std::stoi(v); }
        else if (k == "--huber_delta") { next(); huber_delta = std::stod(v); }
        else if (k == "--prefilter_thresh") { next(); prefilter_thresh = std::stod(v); }
        else if (k == "--min_track_len") { next(); min_track_len = std::stoi(v); }
        else if (k == "--img_w") { next(); img_w = std::stoi(v); }
        else if (k == "--img_h") { next(); img_h = std::stoi(v); }
        else if (k == "--zmin")  { next(); zmin = std::stod(v); }
        else if (k == "--cam_k_mode_map") { next(); cam_k_mode_map = v; }
    }

    std::cout << "[INFO] input  = " << input_path  << "\n"
              << "[INFO] output = " << output_path << "\n"
              << "[INFO] two_stage=" << two_stage
              << " fix_cam_prefix=" << fix_cam_prefix
              << " huber=" << huber_delta
              << " prefilter=" << prefilter_thresh
              << " zmin=" << zmin
              << " stageB_free_t=" << stageB_free_t
              << "\n";

    // 读取输入
    json j = load_json(input_path);
    Eigen::Matrix3d K;
    K << j["K"][0][0], j["K"][0][1], j["K"][0][2],
         j["K"][1][0], j["K"][1][1], j["K"][1][2],
         j["K"][2][0], j["K"][2][1], j["K"][2][2];
    double fx = K(0,0), fy = K(1,1), cx = K(0,2), cy = K(1,2);

    auto cams_json = j["cameras"];
    auto pts_json  = j["points"];
    auto obs_json  = j["observations"];
    const int num_cams = (int)cams_json.size();
    const int num_pts  = (int)pts_json.size();
    const int num_obs  = (int)obs_json.size();
    std::cout << "[INFO] cams=" << num_cams << " pts=" << num_pts << " obs=" << num_obs << "\n";

    // 参数容器
    std::vector<Eigen::VectorXd> cams(num_cams, Eigen::VectorXd::Zero(6));
    std::vector<Eigen::Vector3d> pts(num_pts,  Eigen::Vector3d::Zero());

    for (auto& c : cams_json) {
        int cid = c["id"];
        Eigen::Matrix3d R; Eigen::Vector3d t;
        for (int r=0;r<3;++r)
            for (int c2=0;c2<3;++c2)
                R(r,c2) = c["R_w2c"][r][c2];
        t << c["t_w2c"][0], c["t_w2c"][1], c["t_w2c"][2];
        Eigen::Vector3d aa = RotMatToAngleAxis(R);
        cams[cid].segment<3>(0) = aa;
        cams[cid].segment<3>(3) = t;
    }
    for (auto& p : pts_json) {
        int pid = p["id"];
        pts[pid] << p["X"][0], p["X"][1], p["X"][2];
    }

    // 轨长
    std::vector<int> track_len(num_pts, 0);
    for (auto& o : obs_json) {
        int pid = o["pt_id"];
        if (pid >= 0 && pid < num_pts) track_len[pid]++;
    }

    // 方向映射
    std::unordered_map<int, Mode> cam_mode;
    if (!cam_k_mode_map.empty()) {
        std::ifstream fm(cam_k_mode_map);
        if (fm.is_open()) {
            json jfix; fm >> jfix;
            for (auto it = jfix.begin(); it != jfix.end(); ++it)
                cam_mode[std::stoi(it.key())] = ParseMode(it.value().get<std::string>());
            std::cout << "[INFO] loaded cam_k_mode_map for " << cam_mode.size() << " cameras\n";
        }
    }

    auto predict_uv = [&](int cid, const Eigen::Vector3d& X)->Eigen::Vector2d{
        double aa[3] = {cams[cid][0], cams[cid][1], cams[cid][2]};
        double t[3]  = {cams[cid][3], cams[cid][4], cams[cid][5]};
        double Xw[3] = {X[0], X[1], X[2]};
        double Xc[3];
        ceres::AngleAxisRotatePoint(aa, Xw, Xc);
        Xc[0]+=t[0]; Xc[1]+=t[1]; Xc[2]+=t[2];
        return { fx * Xc[0]/Xc[2] + cx, fy * Xc[1]/Xc[2] + cy };
    };

    // 构图函数
    auto build_problem = [&](bool points_only, bool free_t)->ceres::Problem {
        ceres::Problem problem;

        // 注册参数块
        for (int cid = 0; cid < num_cams; ++cid)
            problem.AddParameterBlock(cams[cid].data(), 6);
        for (int pid = 0; pid < num_pts; ++pid)
            problem.AddParameterBlock(pts[pid].data(), 3);

        int used = 0;
        for (auto& o : obs_json) {
            int cid = o["cam_id"];
            int pid = o["pt_id"];
            if (track_len[pid] < min_track_len) continue;

            double u = o["uv"][0], v = o["uv"][1];
            if (cam_mode.count(cid)) {
                double uf, vf; RemapObsToK(img_w, img_h, cam_mode[cid], u, v, uf, vf);
                u = uf; v = vf;
            }

            // 简单正深度 + zmin
            Eigen::Matrix3d R = AngleAxisToRotMat(cams[cid].segment<3>(0));
            Eigen::Vector3d t = cams[cid].segment<3>(3);
            Eigen::Vector3d Xc = R * pts[pid] + t;
            if (Xc(2) <= zmin) continue;

            // 预过滤
            if (prefilter_thresh > 0.0) {
                Eigen::Vector2d uvp = predict_uv(cid, pts[pid]);
                double e = (uvp - Eigen::Vector2d(u,v)).norm();
                if (e > prefilter_thresh) continue;
            }

            ceres::CostFunction* cost = ReprojError::Create(u, v, fx, fy, cx, cy, zmin);
            // 每条 residual 独立 HuberLoss，避免共享导致释放崩溃
            ceres::LossFunction* loss_this = new ceres::HuberLoss(huber_delta);
            problem.AddResidualBlock(cost, loss_this, cams[cid].data(), pts[pid].data());
            used++;
        }

        std::cout << "[INFO] residuals added: " << used
                  << " | points_only=" << (points_only?1:0)
                  << " | free_t="      << (free_t?1:0)
                  << "\n";

        // 约束设置
        if (points_only) {
            for (int cid = 0; cid < num_cams; ++cid)
                problem.SetParameterBlockConstant(cams[cid].data());
        } else {
            // 固定前缀相机（6DoF 全固定）
            int fixN = std::min(fix_cam_prefix, num_cams);
            for (int cid = 0; cid < fixN; ++cid)
                problem.SetParameterBlockConstant(cams[cid].data());

            // 对其余相机：若 free_t==0，只优化旋转，平移用上下界锁死
            if (!free_t) {
                for (int cid = fixN; cid < num_cams; ++cid) {
                    double* p = cams[cid].data();
                    for (int k = 3; k < 6; ++k) {
                        problem.SetParameterLowerBound(p, k, p[k]);
                        problem.SetParameterUpperBound(p, k, p[k]);
                    }
                }
            }
        }
        return problem;
    };

    // Solver 配置
    ceres::Solver::Options opts;
    opts.linear_solver_type = ceres::SPARSE_SCHUR;
    opts.minimizer_progress_to_stdout = true;
    opts.max_num_iterations = 50;
    opts.num_threads = 8;

    ceres::Solver::Summary summary;

    // Stage-A：仅点
    if (two_stage) {
        std::cout << "[MODE] Stage-A: optimize points only (all cameras fixed)\n";
        ceres::Problem pa = build_problem(/*points_only=*/true, /*free_t=*/false);
        ceres::Solve(opts, &pa, &summary);
        std::cout << "===== CERES SUMMARY (Stage-A) =====\n" << summary.BriefReport() << "\n";

        // 导出 _A.json
        json outA; outA["K"] = j["K"];
        {
            json jc = json::array();
            for (int i = 0; i < num_cams; ++i) {
                Eigen::Vector3d aa = cams[i].segment<3>(0);
                Eigen::Vector3d t  = cams[i].segment<3>(3);
                Eigen::Matrix3d R  = AngleAxisToRotMat(aa);
                jc.push_back({
                    {"id", i},
                    {"R_w2c", {{R(0,0),R(0,1),R(0,2)},
                               {R(1,0),R(1,1),R(1,2)},
                               {R(2,0),R(2,1),R(2,2)}}},
                    {"t_w2c", {t[0], t[1], t[2]}}
                });
            }
            outA["cameras_optimized"] = jc;
        }
        {
            json jp = json::array();
            for (int i = 0; i < num_pts; ++i)
                jp.push_back({{"id", i}, {"X", {pts[i][0], pts[i][1], pts[i][2]}}});
            outA["points_optimized"] = jp;
        }
        outA["notes"] = {
            {"stage","A"},
            {"solver","Ceres BA (points-only)"},
            {"fix_cam_prefix",fix_cam_prefix},
            {"huber_delta_px",huber_delta},
            {"prefilter_thresh_px",prefilter_thresh},
            {"min_track_len",min_track_len},
            {"zmin",zmin},
            {"num_cameras",num_cams},
            {"num_points",num_pts},
            {"num_observations",num_obs},
            {"convention","x_cam = R_w2c * x_world + t_w2c"}
        };
        std::string outA_path = output_path;
        auto pos = outA_path.rfind(".json");
        if (pos != std::string::npos) outA_path = outA_path.substr(0,pos) + "_A.json";
        else outA_path += "_A.json";
        save_json(outA_path, outA);
        std::cout << "[OK] wrote Stage-A to " << outA_path << "\n";
    }

    // Stage-B：全量（可只旋转）
    std::cout << "[MODE] Stage-B: full BA (guarded). stageB_free_t=" << stageB_free_t << "\n";
    ceres::Problem pb = build_problem(/*points_only=*/false, /*free_t=*/(stageB_free_t!=0));
    ceres::Solve(opts, &pb, &summary);
    std::cout << "===== CERES SUMMARY (Stage-B) =====\n" << summary.BriefReport() << "\n";

    // 输出最终
    json out; out["K"] = j["K"];
    {
        json jc = json::array();
        for (int i = 0; i < num_cams; ++i) {
            Eigen::Vector3d aa = cams[i].segment<3>(0);
            Eigen::Vector3d t  = cams[i].segment<3>(3);
            Eigen::Matrix3d R  = AngleAxisToRotMat(aa);
            jc.push_back({
                {"id", i},
                {"R_w2c", {{R(0,0),R(0,1),R(0,2)},
                           {R(1,0),R(1,1),R(1,2)},
                           {R(2,0),R(2,1),R(2,2)}}},
                {"t_w2c", {t[0], t[1], t[2]}}
            });
        }
        out["cameras_optimized"] = jc;
    }
    {
        json jp = json::array();
        for (int i = 0; i < num_pts; ++i)
            jp.push_back({{"id", i}, {"X", {pts[i][0], pts[i][1], pts[i][2]}}});
        out["points_optimized"] = jp;
    }
    out["notes"] = {
        {"stage","B"},
        {"solver","Ceres BA (Schur)"},
        {"two_stage",two_stage},
        {"fix_cam_prefix",fix_cam_prefix},
        {"stageB_free_t",stageB_free_t},
        {"huber_delta_px",huber_delta},
        {"prefilter_thresh_px",prefilter_thresh},
        {"min_track_len",min_track_len},
        {"zmin",zmin},
        {"num_cameras",num_cams},
        {"num_points",num_pts},
        {"num_observations",num_obs},
        {"convention","x_cam = R_w2c * x_world + t_w2c"}
    };
    save_json(output_path, out);
    std::cout << "[OK] wrote refined BA result to " << output_path << "\n";
    return 0;
}
