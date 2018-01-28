#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <queue>
#include <map>
#include "camera.h"

using namespace std;

struct SuperPixel;
class ImgData;

struct Vertex
{
	int nLabel;
	float fWeight;	//The weight of point various
	bool operator<(const Vertex &v) const;	//For priority queue
	Vertex(int label, float weight = 0.0);
	Vertex() {}
};

struct Edge
{
	float fDistance;	//The weight of edge
	Vertex vStart;
	Vertex vEnd;
	Edge(const Vertex &start, const Vertex &end, float dist);
};


struct Graph
{
	ImgData & rData;
	vector<Vertex> vecVertecies;
	vector<vector<Edge>> vecEdges;
	vector<bool> bVisited;
	
	vector<int> Dijkstra(int nStart);
	void Reset();
	Graph(ImgData &img);
	vector<int> GetNeighbour(int nStart);
};

struct SuperPixel
{
	cv::Point ptCentre;
	vector<cv::Point> vecContours;
	vector<cv::Point> vecPixels;
	vector<float> vecDepth;

	unsigned long long uPixelCnt;
	unsigned long long uDepthCnt;
	float fDepthAvr;
	int nLabel;	//Label
	
	void MakeSuperPixel();
	bool IsComplete() const;
	
};

class ImgData
{
public:
	//变量
	int id;
	Camera cam;
	cv::Mat matDepth;
	cv::Mat world_center;
	cv::Mat matImgOrigin;
	cv::Mat matLabels;
	cv::Mat sp_contour;
	int nSuperPixelCnt;
	vector<SuperPixel> vecSPixel;
	string path_output;
	//函数
	ImgData(int _id, Camera& _cam, cv::Mat& _origin_img, cv::Mat& _depth_mat, cv::Mat& _sp_label, cv::Mat& _sp_contour, int _sp_num);
	ImgData() {}
private:
	void CreatePath();
	void calc_world_center();
	bool IsSkyColour(const cv::Point& ptC);
	void SetupSuperPixel();
	void MakeSky();
	void save_depth_image();
	void save_sp_image();
	
	void DepthSynthesize();		//Core
	void ComputeHistogram(vector<cv::Mat>& vecHistos, int nLabel, const SuperPixel &rsp);	//Generate histograms
	bool IsAdjacent(SuperPixel &sp1, SuperPixel & sp2);		//Tell two superpixels' adjacency
};

void mix_pic(std::vector<ImgData>& imgdata_vec, Camera& now_cam, std::vector<int>& img_id, cv::Mat& output_img);


