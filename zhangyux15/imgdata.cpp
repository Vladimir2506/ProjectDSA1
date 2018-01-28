#include "imgdata.h"

#include "global_var.h"
#include "my_math_tool.h"

#include <io.h>  
#include <direct.h>
#include <thread>
#include <queue>
#include <algorithm>
#include <string>
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigen>


using namespace std;
using namespace cv;

ImgData::ImgData(int _id, Camera& _cam, Mat& _origin_img, Mat& _depth_mat, Mat& _sp_label, Mat& _sp_contour, int _sp_num)
{
	cout << "--Init ImgData with file" << to_string(_id) << "..." << endl;			// 不需要重新计算超像素分割
	id = _id;
	cam = _cam;
	nSuperPixelCnt = _sp_num;
	_sp_label.copyTo(matLabels);
	_sp_contour.copyTo(sp_contour);
	_origin_img.copyTo(matImgOrigin);
	_depth_mat.copyTo(matDepth);
	path_output = PATH_MY_OUTPUT + "\\" + to_string(id);
	CreatePath();
	calc_world_center();
	
	SetupSuperPixel();
	//MakeSky(); //Failed
	//DepthSynthesize();

	save_depth_image();
	save_sp_image();
	
}

void SuperPixel::MakeSuperPixel()
{
	uPixelCnt = vecPixels.size();
	Point ptSum(0, 0);
	float fDepthSum = 0;
	uDepthCnt = 0;

	for (unsigned int i = 0; i < uPixelCnt; ++i)
	{
		ptSum += vecPixels[i];
		float fDepth = vecDepth[i];
		if (fDepth > 1e-6)
		{
			fDepthSum += fDepth;
			++uDepthCnt;
		}
	}

	ptCentre = ptSum / int(uPixelCnt);

	if (uDepthCnt == 0)
		fDepthAvr = 0;
	else
		fDepthAvr = fDepthSum / uDepthCnt;

	Mat matMask = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	for (int i = 0; i < vecPixels.size(); i++)
	{
		matMask.at<uchar>(vecPixels[i]) = 1;	
	}
	
	vector<vector<Point>> temp_contour;
	findContours(matMask, temp_contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vecContours = temp_contour[0];	
}

bool SuperPixel::IsComplete() const
{
	return 1.0 * uDepthCnt / uPixelCnt > 0.05;
}

void ImgData::save_depth_image()
{
	cout << "--save_depth_image..." << std::flush;
	for (auto &sp : vecSPixel)
	{
		for (unsigned int i = 0; i < sp.uPixelCnt; ++i)
		{
			matDepth.at<float>(sp.vecPixels[i]) = sp.vecDepth[i];
		}
	}
	// 用彩色表示深度图
	Mat hue_mat;		// 映射到色相0~360
	normalize(matDepth, hue_mat, 255.0, 0, NORM_MINMAX);

	Mat hsv_pic(HEIGHT, WIDTH, CV_8UC3);
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			Vec3b color{ unsigned char(int(hue_mat.at<float>(Point(x, y)))), 100, 255 };
			hsv_pic.at<Vec3b>(Point(x, y)) = color;
		}
	}
	cvtColor(hsv_pic, hsv_pic, CV_HSV2BGR);			// 转换为BGR空间
	imwrite(path_output + "\\depth_map.png", hsv_pic);
	cout << "OK" << endl;
}

void ImgData::save_sp_image()
{
	cout << "--save_superpixel_image..." << std::flush;
	Mat sp_img = matImgOrigin.clone();
	sp_img.setTo(Scalar(255, 255, 255), sp_contour);
	imwrite(path_output + "\\superpixel.png", sp_img);
	cout << "OK" << endl;
}

void ImgData::DepthSynthesize()
{
	//Build graph based on the image
	Graph g(*this);
	auto vecHistogram = vector<Mat>(nSuperPixelCnt, Mat());

	//For all Superpixels compute the histos
	for (int i = 0; i < nSuperPixelCnt; ++i)
	{
		ComputeHistogram(vecHistogram, i, vecSPixel[i]);
	}
	
	//Compute weight of every edge
	for (int i = 0; i < nSuperPixelCnt - 1; ++i)
	{
		for (int j = i + 1; j < nSuperPixelCnt; ++j)
		{
			if (IsAdjacent(vecSPixel[i], vecSPixel[j]))
			{
				float dist = float(compareHist(vecHistogram[i], vecHistogram[j], HISTCMP_CHISQR));
				g.vecEdges[i].emplace_back(i, j, dist);
				g.vecEdges[j].emplace_back(j, i, dist);
			}
		}
	}

	//Run Dijkstra for every superpixel without depth
	for (int i = 0; i < nSuperPixelCnt; ++i)
	{
		//If depth is sufficient
		if (vecSPixel[i].IsComplete()) continue;

		//Perform one dijkstra
		vector<int> vecNeighbour(g.GetNeighbour(i));

		//Load new depth by average method
		float fSumDepth = 0.0;
		float fCntDepth = 0.0;
		for (int j = 0; j < vecNeighbour.size(); ++j)
		{
			fSumDepth += vecSPixel[vecNeighbour[j]].fDepthAvr * vecSPixel[vecNeighbour[j]].uDepthCnt;
			fCntDepth += vecSPixel[vecNeighbour[j]].uDepthCnt;
		}
		float fNewDepth = fSumDepth / fCntDepth;
		vecSPixel[i].fDepthAvr = fNewDepth;
		vecSPixel[i].uDepthCnt = vecSPixel[i].uPixelCnt;
		for (auto &d : vecSPixel[i].vecDepth)
		{
			if (d < 1e-6) d = fNewDepth;
		}
	}
}

void ImgData::ComputeHistogram(vector<cv::Mat>& vecHistos, int nLabel, const SuperPixel & rsp)
{
	//Find the bounding rectangle
	Rect rectEnclosure(boundingRect(rsp.vecContours));
	Mat matImgLab;
	Mat matImgRegion(matImgOrigin(rectEnclosure).clone());
	cvtColor(matImgRegion, matImgLab, CV_BGR2Lab);
	Mat matMask(Mat::zeros(rectEnclosure.height, rectEnclosure.width, CV_8UC1));
	for (auto &p : rsp.vecPixels)
	{
		matMask.at<uchar>(p - rectEnclosure.tl()) = 1;
	}
	
	//From mask get histogram
	Mat matHist3d;
	int arrHistSize[]{ 8,28,28 };
	const int arrChannel[]{ 0,1,2 };
	float arrRange[]{ 0.0,256.0 };
	const float *arrRanges[]{ arrRange,arrRange,arrRange };

	//To compute histogram
	calcHist(&matImgLab, 1, arrChannel, matMask, matHist3d, 3, arrHistSize, arrRanges);
	normalize(matHist3d, vecHistos[nLabel], 100, 0, CV_L1, CV_32F);
}

bool ImgData::IsAdjacent(SuperPixel & sp1, SuperPixel & sp2)
{
	//Find the middle point of two superpixels
	auto lbl1(matLabels.at<int>(sp1.ptCentre));
	auto lbl2(matLabels.at<int>(sp2.ptCentre));
	auto ptMid((sp1.ptCentre + sp2.ptCentre) / 2);

	//If the middle point is in the region of one of the twos
	auto lblMid(matLabels.at<int>(ptMid));
	return lblMid == lbl1 || lblMid == lbl2;
}

void ImgData::CreatePath()
{
	cout << "--CreatePath " << path_output << "..." << std::flush;
	if (_access(path_output.c_str(), 0) == 0)
	{
		string command = "rd /s/q " + path_output;
		system(command.c_str());
	}
	_mkdir(path_output.c_str());
	cout << "OK" << endl;
}

void ImgData::SetupSuperPixel()
{
	// 把每个点加入到超像素对象中
	vecSPixel.resize(nSuperPixelCnt);
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			int nlblSP = matLabels.at<int>(Point(x, y));
			vecSPixel[nlblSP].vecPixels.push_back(Point(x, y));
			float de = matDepth.at<float>(Point(x, y));
			vecSPixel[nlblSP].vecDepth.push_back(de);
		}
	}

	for (int i = 0; i < vecSPixel.size(); i++)
	{
		vecSPixel[i].MakeSuperPixel();
		vecSPixel[i].nLabel = i;
	}
}

void ImgData::MakeSky()
{
	for (auto &sp : vecSPixel)
	{
		if (IsSkyColour(sp.ptCentre))
		{
			if (sp.IsComplete()) continue;
			sp.fDepthAvr = FLT_MAX;
			sp.vecDepth = vector<float>(sp.uPixelCnt, FLT_MAX);
			sp.uDepthCnt = sp.uPixelCnt;
		}
	}
}


void ImgData::calc_world_center()
{
	Mat temp = Mat::zeros(3, 1, CV_32F);
	int nDepthCnt = 0;
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			float depth = matDepth.at<float>(Point(x, y));
			if (depth > 1e-6)
			{
				temp += cam.get_world_pos(Point(x, y), depth);
				nDepthCnt++;
			}
		}
	}
	world_center = temp / nDepthCnt;
}

bool ImgData::IsSkyColour(const cv::Point & ptC)
{
	auto colour = matImgOrigin.at<Vec3b>(ptC);
	auto B = colour[0], G = colour[1], R = colour[2];
	return B - G > 25 && B - R > 25;
}


void shape_preserve_warp(ImgData& imgdata, Camera& novel_cam, Mat& output_img, int thread_rank)
{
	cout << "--thread--" << thread_rank << "--begin shape_preserve_warp..." << endl;
	output_img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	Mat wrap_img_depth = Mat::zeros(HEIGHT, WIDTH, CV_32F);			//记录wrap后img的深度图
	Mat reproject_mat, reproject_vec;
	imgdata.cam.fill_reprojection(novel_cam, reproject_mat, reproject_vec);

	// 计算每个超像素在新视点下的深度
	vector<float> depth_dict(imgdata.nSuperPixelCnt);
	for (int i = 0; i < imgdata.nSuperPixelCnt; i++)
	{
		SuperPixel& superpixel = imgdata.vecSPixel[i];
		
		Mat temp_mat = novel_cam.cam_pos - imgdata.cam.get_world_pos(superpixel.ptCentre, superpixel.fDepthAvr);
		depth_dict[i] = sqrt(temp_mat.dot(temp_mat));
	}

	clock_t t1 = 0;


	// 逐个超像素进行warp
	for (int i = 0; i < imgdata.nSuperPixelCnt; i++)
	{
		SuperPixel& superpixel = imgdata.vecSPixel[i];

		vector<Point2f> triangle;
		minEnclosingTriangle(superpixel.vecPixels, triangle);			// 计算最小外接三角形
																	//calculate ep
		Eigen::Matrix<float, 6, 6> ep_mat_eigen = Eigen::Matrix<float, 6, 6>::Zero();
		Eigen::Matrix<float, 6, 1 >ep_vec_eigen = Eigen::Matrix<float, 6, 1>::Zero();

		Eigen::MatrixXf A_mat = Eigen::MatrixXf::Zero(superpixel.uPixelCnt * 2, 6);
		Eigen::MatrixXf b_mat = Eigen::MatrixXf::Zero(superpixel.uPixelCnt * 2, 1);


		vector<float> coefficient(3);
		{
			for (unsigned int j = 0; j < superpixel.uPixelCnt; j++)
			{
				Point& origin_point = superpixel.vecPixels[j];
				float point_depth = superpixel.vecDepth[j];
				// 检验是否有深度
				if (point_depth < 1e-6)
					continue;

				// 插值到外接三角形里，得到用三个定点表示的参数
				tri_interpolation(triangle, origin_point, coefficient);

				// 计算在新视点下的像素坐标	
				Point destination_point = cal_reprojection(origin_point, point_depth, reproject_mat, reproject_vec);
				for (int i = 0; i < 3; i++) A_mat(2 * j, i) = A_mat(2 * j + 1, i + 3) = coefficient[i % 3];
				b_mat(2 * j, 0) = destination_point.x;
				b_mat(2 * j + 1, 0) = destination_point.y;
			}
		}
		ep_mat_eigen = A_mat.transpose()*A_mat;
		ep_vec_eigen = A_mat.transpose()*b_mat;

		// 计算es_mat，衡量三角形的形变量
		Eigen::Matrix<float, 6, 6> es_mat_eigen = Eigen::Matrix<float, 6, 6>::Zero();
		{
			int j, k, l;
			for (int iter_time = 0; iter_time < 3; iter_time++)
			{
				switch (iter_time)
				{
				case 0:
					j = 0; k = 1; l = 2;
					break;
				case 1:
					j = 1; k = 2; l = 0;
					break;
				case 2:
					j = 2; k = 0; l = 1;
					break;
				default:
					break;
				}

				Point2f& pj = triangle[j]; float xj = pj.x, yj = pj.y;
				Point2f& pk = triangle[k]; float xk = pk.x, yk = pk.y;
				Point2f& pl = triangle[l]; float xl = pl.x, yl = pl.y;

				Eigen::Vector2f p1(xk, yk);
				Eigen::Vector2f p2(xj, yj);
				Eigen::Vector2f p3(xl, yl);

				Eigen::Matrix2f R90;
				R90 <<
					0, 1,
					-1, 0;

				Eigen::Matrix<float, 2, 6> A = Eigen::Matrix<float, 2, 6>::Zero();
				float a = (p3 - p1).dot(p2 - p1) / (p2 - p1).squaredNorm();
				float b = (p3 - p1).dot(R90*(p2 - p1)) / (p2 - p1).squaredNorm();

				//E=|p3-p1-a(p2-p1)-bR90(p2-p1)|^2
				Eigen::Matrix2f Aj, Ak, Al;
				Ak <<
					(-1 + a), b,
					-b, (-1 + a);
				Aj <<
					-a, -b,
					b, -a;
				Al <<
					1, 0,
					0, 1;
				A.col(j) = Aj.col(0);
				A.col(j + 3) = Aj.col(1);
				A.col(k) = Ak.col(0);
				A.col(k + 3) = Ak.col(1);
				A.col(l) = Al.col(0);
				A.col(l + 3) = Al.col(1);

				Eigen::Matrix<float, 6, 6 >ATA = A.transpose()*A;
				es_mat_eigen += ATA;
			}
		}
		// 求逆矩阵，计算在新视点下的外接三角形
		float es_weight = 1;

		Eigen::Matrix<float, 6, 6> temp_mat_eigen = ep_mat_eigen + es_mat_eigen * es_weight;
		if (temp_mat_eigen.determinant() < 1e-6) continue;
		Eigen::LDLT<Eigen::MatrixXf> ldlt(temp_mat_eigen);
		Eigen::VectorXf x = ldlt.solve(ep_vec_eigen);

		vector<Point2f> novel_triangle;
		novel_triangle.resize(3);
		novel_triangle[0] = Point2f(x(0), x(3));
		novel_triangle[1] = Point2f(x(1), x(4));
		novel_triangle[2] = Point2f(x(2), x(5));
		//如果面积之比大于4，则跳过
		float origin_area = calc_triangle_area(triangle);
		float new_area = calc_triangle_area(novel_triangle);
		if (new_area / origin_area > 4)
			continue;

		// 把原来超像素的轮廓用三角形插值投影到新视点下
		vector<Point> novel_contour(superpixel.vecContours.size());

		vector<Point> novel_points;
		for (int j = 0; j < superpixel.vecContours.size(); j++)
		{
			Point& origin_point = superpixel.vecContours[j];
			tri_interpolation(triangle, origin_point, coefficient);
			novel_contour[j] = (inv_tri_interpolation(novel_triangle, coefficient));
		}
		// 用投影后的轮廓得到投影后的超像素区域
		contour_to_set(novel_contour, novel_points);

		for (int j = 0; j < novel_points.size(); j++)
		{
			Point& novel_point = novel_points[j];
			vector<float> coefficient;
			tri_interpolation(novel_triangle, novel_point, coefficient);
			Point reproject_point = inv_tri_interpolation(triangle, coefficient);
			if (check_range(reproject_point) && check_range(novel_point))
			{
				float& before_depth = wrap_img_depth.at<float>(novel_point);
				if (abs(before_depth - 0) < 1e-6 || depth_dict[i] < before_depth)
				{
					before_depth = depth_dict[i];
					output_img.at<Vec3b>(novel_point) = imgdata.matImgOrigin.at<Vec3b>(reproject_point);
				}
			}
		}
	}
	cout << "--thread--" << thread_rank << "--end " << endl;
}

void mix_pic(vector<ImgData>& imgdata_vec, Camera& now_cam, vector<int>& img_id, Mat& output_img)
{
	cout << "--begin generate pic..." << endl;
	clock_t start;
	clock_t end;
	start = clock();
	output_img = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
	vector<Mat> warp_img(img_id.size());
	vector<thread> threads(img_id.size());
	for (int i = 0; i < img_id.size(); i++)
	{
		//对四个最近的照片分别warp到同一个视点
		threads[i] = thread(shape_preserve_warp, ref(imgdata_vec[img_id[i]]), ref(now_cam), ref(warp_img[i]), i);
	}
	//等待所有线程执行完毕
	for (int i = 0; i < threads.size(); i++)
	{
		threads[i].join();
	}
	//图像融合
	for (int x = 0; x < WIDTH; x++)
	{
		for (int y = 0; y < HEIGHT; y++)
		{
			Point point(x, y);
			for (int i = 0; i < img_id.size(); i++)
			{
				//按照片远近优先级赋值
				if (warp_img[i].at<Vec3b>(point) != Vec3b{ 0,0,0 })
				{
					output_img.at<Vec3b>(point) = warp_img[i].at<Vec3b>(point);
					break;		//一旦找到一张warp后图像有图像信息，则赋值完跳出循环
				}
			}
		}
	}

	end = clock();
	cout << "--OK using time:" << (end - start) << "ms" << endl;

}

bool Vertex::operator<(const Vertex & v) const
{
	return fWeight > v.fWeight;
}

Vertex::Vertex(int label, float weight)
{
	nLabel = label;
	fWeight = weight;
}

Graph::Graph(ImgData &img)
	:rData(img)
{
	//Adjacent matrix
	vecEdges = vector<vector<Edge>>(img.nSuperPixelCnt, vector<Edge>());
	vecVertecies = vector<Vertex>(img.nSuperPixelCnt);
	//Visit array
	bVisited = vector<bool>(img.nSuperPixelCnt);
	for (int i = 0; i < img.nSuperPixelCnt; ++i)
	{
		vecVertecies.emplace_back(i);
	}
}

vector<int> Graph::GetNeighbour(int nStart)
{
	Reset();
	return Dijkstra(nStart);
}


vector<int> Graph::Dijkstra(int nStart)
{
	vector<int> vecAnswer;
	vector<double> vecDist = vector<double>(rData.nSuperPixelCnt, FLT_MAX);
	map<double, int> mapCandidate;
	priority_queue<Vertex> pqDij;
	
	//Take points from priority queue

	pqDij.push(nStart);
	while (!pqDij.empty())
	{
		Vertex v(pqDij.top());
		pqDij.pop();
		if (bVisited[v.nLabel]) continue;

		//Collect 40~60 target superpixels
		if (rData.vecSPixel[v.nLabel].IsComplete())
		{
			mapCandidate.emplace(vecDist[v.nLabel], v.nLabel);
			if (mapCandidate.size() > MAX_DIJKSTRA_VERTECIES) break;
		}

		//Iterations for Dijkstra
		for (int i = 0; i < vecEdges[v.nLabel].size(); ++i)
		{
			Vertex u = vecEdges[v.nLabel][i].vEnd;
			if (vecDist[u.nLabel] > v.fWeight + vecEdges[v.nLabel][i].fDistance)
			{
				vecDist[u.nLabel] = v.fWeight + vecEdges[v.nLabel][i].fDistance;
				pqDij.push(u);
			}
		}
	}

	//Select 4~6 out of candidates
	auto it = mapCandidate.begin();
	for (int i = 0; i < MAX_SAMPLE_COMPLEMENT; ++i)
	{
		vecAnswer.emplace_back((it++)->second);
	}
	return vecAnswer;
}

void Graph::Reset()
{
	//Reset weights and status of the graph
	for (auto & v : vecVertecies)
	{
		v.fWeight = 0.0;
	}
	bVisited = vector<bool>(rData.nSuperPixelCnt, false);
}


Edge::Edge(const Vertex & start, const Vertex & end, float dist)
	:vStart(start), vEnd(end)
{	
	fDistance = dist;
}
