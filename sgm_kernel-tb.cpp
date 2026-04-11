#include "sgm_kernel-tb.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

static inline bool file_exist(const std::string &p)
{
	return !p.empty();
}

int main(int argc, char** argv)
{

    std::string left_path  = (argc > 1) ? argv[1] :
    		"/home/hamzy/SGM/Kitti-Data/training/image_2/000000_10.png";
    std::string right_path = (argc > 2) ? argv[2] :
    		"/home/hamzy/SGM/Kitti-Data/training/image_3/000000_10.png";

    std::string gt_path = (argc > 3) ? argv[3] :
        	"/home/hamzy/SGM/Kitti-Data/training/disp_noc_0/000000_10.png";

    if (!file_exist(left_path) || !file_exist(right_path) || !file_exist(gt_path))
    {
    	std::cerr << "ERROR: Provide left & right images and the ground truth"
    			<< " left: " << left_path << "\n"
				<< "right: " << right_path << "\n"
				<< "gt: " << gt_path << std::endl;
    	return 1;
    }
    /* Load as grayscale */
    cv::Mat left  = cv::imread(left_path,  cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
    cv::Mat gt = cv::imread(gt_path, cv::IMREAD_UNCHANGED);
    if (left.empty() || right.empty() || gt.empty())
    {
        std::cerr << "ERROR: Could not load input images:\n  "
                  << left_path << "\n  " << right_path <<  "\n  " << gt_path << std::endl;
        return 2;
    }

    /* Ensure size matches kernel shape */
    if (left.cols != IMG_W || left.rows != IMG_H)
    {
        cv::resize(left,  left,  cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_AREA);
    }
    if (right.cols != IMG_W || right.rows != IMG_H)
    {
        cv::resize(right, right, cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_AREA);
    }

    const float scale_x = static_cast<float>(IMG_W) / static_cast<float>(gt.cols);

    cv::Mat gt_f;
    gt.convertTo(gt_f, CV_32F, 1.0f / 256.0f);

    if(gt_f.cols != IMG_W || gt_f.rows != IMG_H)
    {
    	cv::resize(gt_f, gt_f, cv::Size(IMG_W, IMG_H), 0, 0, cv::INTER_NEAREST);
    	gt_f *= scale_x;
    }

    if (left.cols != IMG_W || left.rows != IMG_H ||
        right.cols != IMG_W || right.rows != IMG_H ||
		gt_f.cols != IMG_W || gt_f.rows != IMG_H)
    {
        std::cerr << "ERROR: Size mismatch after resize. "
                  << "Expected (" << IMG_W << "x" << IMG_H << ").\n";
        return 3;
    }

    /* Simulate AXI4-Stream interfaces */
    hls::stream<pix_t> left_stream;
    hls::stream<pix_t> right_stream;
    hls::stream<pix_t> disp_stream;

    for (int r = 0; r < IMG_H; ++r)
    {
        const uint8_t* lp = left.ptr<uint8_t>(r);
        const uint8_t* rp = right.ptr<uint8_t>(r);
        for (int c = 0; c < IMG_W; ++c)
        {
            left_stream.write(static_cast<pix_t>(lp[c]));
            right_stream.write(static_cast<pix_t>(rp[c]));
        }
    }

    /* Run kernel */
    sgm_kernel(left_stream, right_stream, disp_stream);

    /* Retrieve output disparity */
#if (DISP > 256)
    cv::Mat disp(IMG_H, IMG_W, CV_16U);
    using out_u_t = uint16_t;
#else
    cv::Mat disp(IMG_H, IMG_W, CV_8U);
        using out_u_t = uint8_t;
#endif

    const int expected = IMG_W * IMG_H;
    int result = 0;

    for (int r = 0; r < IMG_H; ++r)
    {
#if (DISP > 256)
        uint16_t *dp = disp.ptr<uint16_t>(r);
#else
        uint8_t *dp = disp.ptr<uint8_t>(r);
#endif
        for (int c = 0; c < IMG_W; ++c)
        {
        	if (disp_stream.empty())
        	{
        		std::cerr << "ERROR: disp_stream underrun at pixel "
        				<< result << "/" << expected << std::endl;
        		return 4;
        	}
        	out_u_t v = static_cast<out_u_t>(disp_stream.read());
            dp[c] = v;
            result++;
        }
    }

    if (result != expected)
    {
        std::cerr << "ERROR: expected " << expected << " disparity pixels, resulted "
        		<< result << std::endl;
        return 5;
    }

    double disp_min = 0.0, disp_max = 0.0;
    cv::minMaxLoc(disp, &disp_min, &disp_max);
    std::cout << "disp min=" << disp_min << " max=" << disp_max << "\n";

    int nonzero = 0;
    for (int r = 0; r < IMG_H; ++r)
    {
        for (int c = 0; c < IMG_W; ++c)
        {
            if (disp.at<out_u_t>(r,c) != 0) nonzero++;
        }
    }
    std::cout << "nonzero disparity pixels: " << nonzero
              << " / " << (IMG_W * IMG_H) << "\n";

    int test_r = IMG_H / 2;
    for (int c : {IMG_W/4, IMG_W/2, 3*IMG_W/4})
    {
        std::cout << "disp(" << test_r << "," << c << ") = "
                  << int(disp.at<out_u_t>(test_r, c)) << "\n";
    }

    int count0 = 0, count1to3 = 0, count4plus = 0;
    for (int r = 0; r < IMG_H; ++r) {
        for (int c = 0; c < IMG_W; ++c) {
            int v = int(disp.at<out_u_t>(r,c));
            if (v == 0) count0++;
            else if (v <= 3) count1to3++;
            else count4plus++;
        }
    }
    std::cout << "count0=" << count0
              << " count1to3=" << count1to3
              << " count4plus=" << count4plus << "\n";

    /* Save result */
#if (DISP > 256)
    const char* raw_name = "disp_u16.png";
#else
    const char* raw_name = "disp_u8.png";
#endif
    cv::imwrite(raw_name, disp);
    std::cout << "OK: Disparity map written to " << raw_name << " ("
            << IMG_W << "x" << IMG_H << ")\n";

    /* Visualization */
    double mn = 0, mx = 0;
        cv::minMaxLoc(disp, &mn, &mx);
        double scale_den = (mx > 0) ? mx : std::max(1, DISP - 1);
        cv::Mat disp_vis_8u;
        disp.convertTo(disp_vis_8u, CV_8U, 255.0 / scale_den);

        cv::imwrite("disp_vis.png", disp_vis_8u);
        cv::Mat disp_color;
        cv::applyColorMap(disp_vis_8u, disp_color, cv::COLORMAP_JET);
        cv::imwrite("disp_color.png", disp_color);

        int valid_count = 0, bad1 = 0, bad3 = 0;
        double sum_abs_err = 0.0;
        cv::Mat err_map(IMG_H, IMG_W, CV_32F, cv::Scalar(0));

        for (int r = 0; r < IMG_H; ++r)
        {
        	for (int c = 0; c < IMG_W; ++c)
        	{
        		float gt_disp = gt_f.at<float>(r,c);
        		if(gt_disp <= 0.0f) continue;

        		float est_disp = float(disp.at<out_u_t>(r,c));
        		float err = std::abs(est_disp - gt_disp);

        		err_map.at<float>(r,c) = err;

        		++valid_count;
        		sum_abs_err += err;
        		if (err > 1.0f) bad1++;
        		if (err > 3.0f) bad3++;
        	}
        }

        double err_max = 0.0;
        cv::minMaxLoc(err_map, nullptr, &err_max);

        cv::Mat err_vis;
        double scale = (err_max > 0.0) ? (255.0 / err_max) : 1.0;

        err_map.convertTo(err_vis, CV_8U, scale);

        cv::imwrite("err_map.png", err_vis);

        if (valid_count == 0)
        {
            std::cerr << "ERROR: No valid GT pixels for comparison\n";
            return 6;
        }
        std::cout << "Valid pixels: " << valid_count << "\n";
        std::cout << "MAE: " << (sum_abs_err / valid_count) << "\n";
        std::cout << "Bad >1 px: " << (100.0 * bad1 / valid_count) << "%\n";
        std::cout << "Bad >3 px: " << (100.0 * bad3 / valid_count) << "%\n";

        return 0;
}
