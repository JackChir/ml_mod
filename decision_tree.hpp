/**
 * @file decision_tree.hpp
 * @brief 决策树，包含ID3算法和C4.5算法
 * @details
 * class_classifier_t: 是数据的标签分类器
 * param_classifier_t: 是数据的分类器，能够对数据传入的各个维度进行分类
 */
#ifndef _DECISION_TREE_HPP_
#define _DECISION_TREE_HPP_

#include <vector>
#include <map>
#include <cmath>
#include <tuple>
#include <algorithm>

#include "mat.hpp"

// 求熵
double cal_entropy(const std::vector<double>& samples) {
	double entropy = 0.0;
	double total = 0.0;
	for (double sample : samples) {
		total += sample;
	}
	for (double sample : samples) {
		double p = sample / total;
		entropy -= p * std::log2(p);
	}
	return entropy;
}

template<int dim_size, typename param_t, typename class_classifier_t>
double cal_vec_entropy(const std::vector<mat<dim_size, 1, param_t> >& vdata, class_classifier_t& f_cc)
{
	std::map<int, double> mp;                               // 类别到各类的标签的映射
    // 统计经过类别判别器后的各类的数量
	for (auto itr = vdata.begin(); itr != vdata.end(); ++itr)
	{
		const mat<dim_size, 1, param_t>& mt = *itr;
		int i_class = f_cc(mt[dim_size-1]);
		if (mp.count(i_class) == 0) 
		{
			mp.insert(std::make_pair(i_class, 1));
		}
		else 
		{
			mp[i_class]++;
		}
	}
    // 把数据整理称为用于计算熵值的数组
	std::vector<double> vec_cnts;
	for (auto itr = mp.begin(); itr != mp.end(); ++itr)
	{
		vec_cnts.push_back(itr->second);
	}
	return cal_entropy(vec_cnts);               // 计算所有类别的熵值
}

/* 使用idx的参数分类器对数据进行分类，并计算分类后的熵 */
template<typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
double cal_div_entropy(const std::vector<mat<dim_size, 1, param_t> >& vdata, const int& idx, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
    // 将数据按照idx位置对应的分类器进行分类，划分到不同的类别中
	std::map<int, std::vector<mat<dim_size, 1, param_t> > > mp;             // 经过分类器判别后的各类数量
	for (auto itr = vdata.begin(); itr != vdata.end(); ++itr)
	{
		const mat<dim_size, 1, param_t>& mt = *itr;
		int i_class = f_pc(idx, mt);
		if (mp.count(i_class) == 0) 
		{
			mp.insert(std::make_pair(i_class, std::vector<mat<dim_size, 1, param_t> >()));
		}
		mp[i_class].push_back(mt);
	}
    // 计算每个类别中实际数据类别的熵值，这样根据某分类器分类后，如果分类效果不佳，那么熵值就会很大；反之，如果分类器效果较好，那么熵值就会很小
	double d_all_entropy = 0.;
	for (auto itr = mp.begin(); itr != mp.end(); ++itr) 
	{
		d_all_entropy += (cal_vec_entropy(itr->second, f_cc) * itr->second.size() / vdata.size());
	}
	return d_all_entropy;
}

// 找到分类最佳的分类器，返回这个分类器的索引
template<int pc_size, typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
int max_entropy_gain_index(const std::vector<mat<dim_size, 1, param_t> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc, double& d_max_gain)
{
	double d_all = cal_vec_entropy(vdata, f_cc);
	d_max_gain = -1e10;
	int i_max_idx = 0;
    // 循环数组中的所有分类器，找到使的熵值下降最快的那个
	for (int i = 0; i < pc_size; ++i)
	{
		double d_cur_gain = d_all - cal_div_entropy(vdata, i, f_pc, f_cc);
		if (d_cur_gain > d_max_gain) 
		{
			d_max_gain = d_cur_gain;
			i_max_idx = i;
		}
	}
	return i_max_idx;           // 返回最佳分类器的索引
}

/* 根据idx位置的值分割数据集，返回值为idx位置类别到数据集的映射 */
template<typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
std::map<int, std::vector<mat<dim_size, 1, param_t> > > div_data(const std::vector<mat<dim_size, 1, param_t> >& vdata, const int& idx, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	std::map<int, std::vector<mat<dim_size, 1, param_t> > > mp;
	for (auto itr = vdata.begin(); itr != vdata.end(); ++itr)
	{
		const mat<dim_size, 1, param_t>& mt = *itr;
		int i_class = f_pc(idx, mt);
		if (mp.count(i_class) == 0)
		{
			mp.insert(std::make_pair(i_class, std::vector<mat<dim_size, 1, param_t> >()));
		}
		mp[i_class].push_back(mt);
	}
	return mp;
}

/* 判断类型是不是一类，这里可以修改成满足多少比例的为1类就算是该类 */
template<typename class_classifier_t, typename param_t>
bool same_class(int& i_class, const std::vector<param_t>& vdata,  class_classifier_t& f_cc, double& rate, const double& class_rate=0.9)
{
	i_class = f_cc(vdata[0]);
	if (vdata.size() < 2) 
	{
        rate = 1.;
		return true;
	}
    std::map<int, int> mp;		// 统计类别的数量
	for (int i = 0; i < vdata.size(); ++i) 
	{
		int i_cur_class = f_cc(vdata[i]);
		mp[i_cur_class] = (mp.count(i_cur_class) == 0) ? 1 : mp[i_cur_class] + 1;
	}
    // 找到最大分类的类别
    int i_max_class = -1;
    int i_max_cnt = 0;
    for (auto itr = mp.begin(); itr != mp.end(); ++itr)
    {
        if (itr->second > i_max_cnt) 
        {
            i_max_cnt = itr->second;
            i_max_class = itr->first;
        }
    }
    // 计算正确率
    rate = static_cast<double>(i_max_cnt) / vdata.size();
	return rate >= class_rate;  // 如果正确率大于class_rate，那么就认为是同一类
}

struct dt_node 
{
	bool is_leave;							// 是否为叶子
	int idx;								// 当前使用分类器的索引
	int lbl;								// 当前节点类别
	double rate;							// 正确率
	std::map<int, dt_node*>	mp_sub;			// 下层节点
	dt_node() :is_leave(false), lbl(-1), idx(-1), rate(1.)
	{}
	~dt_node() 
	{
		for (auto itr = mp_sub.begin(); itr != mp_sub.end(); ++itr) 
		{
			delete itr->second;
		}
		mp_sub.clear();
	}
};

template<int pc_size, typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
void _gen_id3_tree(struct dt_node* p_cur_node, const std::vector<mat<dim_size, 1, param_t> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc, const double& class_rate=0.9)
{
	int i_class = 0;
    double f_rate = 0.;
	if (same_class(i_class, vdata, f_cc, f_rate, class_rate)) 
	{
		p_cur_node->lbl = i_class;
		p_cur_node->is_leave = true;
        p_cur_node->rate = f_rate;
		return;
	}
	double d_max_gain = 0.;
	int i_max_idx = max_entropy_gain_index<pc_size>(vdata, f_pc, f_cc, d_max_gain);	// 获取最大分割索引
	// 增加一个保护，如果对于最优的分类器的增益小于1e-10，那么就认为是同一类，这样可以防止陷入死循环，同时设置增益阈值也可以防止过度拟合
	if (d_max_gain < 1e-10) 
	{
		p_cur_node->is_leave = true;
		p_cur_node->lbl = i_class;
		p_cur_node->rate = f_rate;
		return;
	}
	// 计算当前节点的类别
	p_cur_node->is_leave = false;
	p_cur_node->idx = i_max_idx;	// 选择当前分类器
	p_cur_node->lbl = i_class;
	p_cur_node->rate = f_rate;
	std::map<int, std::vector<mat<dim_size, 1, param_t> > > mp_div = div_data(vdata, p_cur_node->idx, f_pc, f_cc);		// 分割数据集
	for (auto itr = mp_div.begin(); itr != mp_div.end(); ++itr)															// 循环判断子集合的决策树
	{
		struct dt_node* p_sub_node = new struct dt_node();																// 创建一个新的节点
		_gen_id3_tree<pc_size>(p_sub_node, itr->second, f_pc, f_cc, class_rate); 												// 生成子数据集的决策树
		p_cur_node->mp_sub.insert(std::make_pair(itr->first, p_sub_node));												// 将子决策树加到当前决策树的下面
	}
}

template<int pc_size, typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
dt_node* gen_id3_tree(const std::vector<mat<dim_size, 1, param_t> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc, const double& class_rate=0.9)
{
	struct dt_node* p_tree = new struct dt_node();
	_gen_id3_tree<pc_size>(p_tree, vdata, f_pc, f_cc, class_rate);
	return p_tree;
}

template<typename param_classifier_t, int dim_size, typename param_t>
std::tuple<int, double> judge_id3(struct dt_node* p_cur_node, const mat<dim_size, 1, param_t>& data, param_classifier_t& f_pc, const int& def_value)
{
	if (p_cur_node->is_leave) 
	{
		return std::tie(p_cur_node->lbl, p_cur_node->rate);
	}
	int i_next_idx = f_pc(p_cur_node->idx, data);
	if (p_cur_node->mp_sub.count(i_next_idx) == 0)			// 之前训练时候没有遇到过的分类
	{
		return std::tuple(def_value, 1.);
	}
	return judge_id3(p_cur_node->mp_sub[i_next_idx], data, f_pc, def_value);
}

/* C4.5 */
// c4.5算法计算分类的时候会根据数据量对分类器进行加权
template<typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
std::tuple<double, double> cal_expect_entropy_and_iv(const std::vector<mat<dim_size, 1, param_t> >& vdata, const int& idx, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	std::map<int, std::vector<mat<dim_size, 1, param_t> > > mp;				// 按照idx进行分类，分类后的类别到数据的映射就是这个
	for (auto itr = vdata.begin(); itr != vdata.end(); ++itr)
	{
		const mat<dim_size, 1, param_t>& mt = *itr;
		int i_class = f_pc(idx, mt);
		if (mp.count(i_class) == 0)
		{
			mp.insert(std::make_pair(i_class, std::vector<mat<dim_size, 1, param_t> >()));
		}
		mp[i_class].push_back(mt);
	}
	double d_all_entropy = 0.;
	double iv = 0.;
	for (auto itr = mp.begin(); itr != mp.end(); ++itr)
	{
		double p = static_cast<double>(itr->second.size()) / vdata.size();
		d_all_entropy += (cal_vec_entropy(itr->second, f_cc) * p);
		iv -= p * std::log2(p);
	}
	return std::make_tuple(d_all_entropy, iv);
}

// 使用c4.5的方法计算最大比例的分类器
template<int pc_size, typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
int max_entropy_gain_ratio_index(const std::vector<mat<dim_size, 1, param_t> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc, double& d_max_gain)
{
	double d_all = cal_vec_entropy(vdata, f_cc);
	d_max_gain = -1e10;
	int i_max_idx = 0;
	for (int i = 0; i < pc_size; ++i)
	{
		double d_cur_entropy = 0., iv = 0.;
		std::tie(d_cur_entropy, iv) = cal_expect_entropy_and_iv(vdata, i, f_pc, f_cc);
		double d_cur_gain = (d_all - d_cur_entropy) / iv;
		if (d_cur_gain > d_max_gain)
		{
			d_max_gain = d_cur_gain;
			i_max_idx = i;
		}
	}
	return i_max_idx;
}

template<int pc_size, typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
void _gen_c45_tree(struct dt_node* p_cur_node, const std::vector<mat<dim_size, 1, param_t> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc, const double& class_rate=0.9)
{
	int i_class = 0;
    double f_rate = 0.;
	if (same_class(i_class, vdata, f_cc, f_rate, class_rate)) 
    {
        p_cur_node->lbl = i_class;
        p_cur_node->is_leave = true;
        p_cur_node->rate = f_rate;
        return;
    }
	double d_max_gain = 0.;
	int i_max_idx = max_entropy_gain_ratio_index<pc_size>(vdata, f_pc, f_cc, d_max_gain);	// 获取最大分割索引
	// 增加一个保护，如果对于最优的分类器的增益小于1e-10，那么就认为是同一类，这样可以防止陷入死循环，同时设置增益阈值也可以防止过度拟合
	if (d_max_gain < 1e-10) 
	{
		p_cur_node->is_leave = true;
		p_cur_node->lbl = i_class;
		p_cur_node->rate = f_rate;
		return;
	}
	p_cur_node->idx = i_max_idx;	// 选择当前分类器
	p_cur_node->lbl = i_class;
	p_cur_node->rate = f_rate;
	p_cur_node->is_leave = false;
	std::map<int, std::vector<mat<dim_size, 1, param_t> > > mp_div = div_data(vdata, p_cur_node->idx, f_pc, f_cc);		// 分割数据集
	for (auto itr = mp_div.begin(); itr != mp_div.end(); ++itr)															// 循环判断子集合的决策树
	{
		struct dt_node* p_sub_node = new struct dt_node();																// 创建一个新的节点
		_gen_c45_tree<pc_size>(p_sub_node, itr->second, f_pc, f_cc, class_rate);																	// 生成子数据集的决策树
		p_cur_node->mp_sub.insert(std::make_pair(itr->first, p_sub_node));												// 将子决策树加到当前决策树的下面
	}
}

template<int pc_size, typename param_classifier_t, typename class_classifier_t, int dim_size, typename param_t>
dt_node* gen_c45_tree(const std::vector<mat<dim_size, 1, param_t> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc, const double& class_rate=0.9)
{
	struct dt_node* p_tree = new struct dt_node();
	_gen_c45_tree<pc_size>(p_tree, vdata, f_pc, f_cc, class_rate);
	return p_tree;
}

template<typename param_classifier_t, int dim_size, typename param_t>
std::tuple<int, double> judge_c45(struct dt_node* p_cur_node, const mat<dim_size, 1, param_t>& data, param_classifier_t& f_pc, const int& def_value)
{
	if (p_cur_node->is_leave)
	{
		return std::tie(p_cur_node->lbl, p_cur_node->rate);
	}
	int i_next_idx = f_pc(p_cur_node->idx, data);
	if (p_cur_node->mp_sub.count(i_next_idx) == 0)			// 之前训练时候没有遇到过的分类
	{
		return std::tuple(def_value, 1.);
	}
	return judge_c45(p_cur_node->mp_sub[i_next_idx], data, f_pc, def_value);
}

#include "ht_memory.h"

void write_file(const dt_node* p_tree, ht_memory& mry)
{
	mry << (p_tree->is_leave?(int)1:(int)0);	// 是否为叶子节点
	mry << p_tree->idx;
	mry << p_tree->lbl;
	mry << p_tree->rate;
	int i_sub_size = static_cast<int>(p_tree->mp_sub.size());
	mry << i_sub_size;		// 子节点的数量
	for (auto itr = p_tree->mp_sub.begin(); itr != p_tree->mp_sub.end(); ++itr) 
	{
		write_file(itr->second, mry);
	}
}
void read_file(ht_memory& mry, dt_node* p_tree)
{
	int is_leave = 0;
	mry >> is_leave;		// 是否为叶子节点
	p_tree->is_leave = (is_leave == 1);
	mry >> p_tree->idx;
	mry >> p_tree->lbl;
	mry >> p_tree->rate;
	int i_sub_size = 0;
	mry >> i_sub_size;		// 子节点的数量
	for (int i = 0; i < i_sub_size; ++i) 
	{
		dt_node* p_sub_node = new dt_node();
		read_file(mry, p_sub_node);
		p_tree->mp_sub.insert(std::make_pair(i, p_sub_node));
	}
}

#endif
