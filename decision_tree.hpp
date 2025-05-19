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

template<int dim_size, typename class_classifier_t>
double cal_vec_entropy(const std::vector<mat<dim_size, 1, double> >& vdata, class_classifier_t& f_cc)
{
	std::map<int, double> mp;
	for (auto itr = vdata.begin(); itr != vdata.end(); ++itr)
	{
		const mat<dim_size, 1, double>& mt = *itr;
		int i_class = f_cc(mt[dim_size-1]);
		mp[i_class] = (mp.count(i_class) == 0 ? 1 : mp[i_class] + 1);
	}
	std::vector<double> vec_cnts;
	for (auto itr = mp.begin(); itr != mp.end(); ++itr)
	{
		vec_cnts.push_back(itr->second);
	}
	return cal_entropy(vec_cnts);
}

/* 计算idx位置分割类的期望熵 */
template<typename param_classifier_t, typename class_classifier_t, int dim_size>
double cal_div_entropy(const std::vector<mat<dim_size, 1, double> >& vdata, const int& idx, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	std::map<int, std::vector<mat<dim_size, 1, double> > > mp;
	for (auto itr = vdata.begin(); itr != vdata.end(); ++itr)
	{
		const mat<dim_size, 1, double>& mt = *itr;
		int i_class = f_pc(idx, mt[idx]);
		if (mp.count(i_class) == 0) 
		{
			mp.insert(std::make_pair(i_class, std::vector<mat<dim_size, 1, double> >()));
		}
		mp[i_class].push_back(mt);
	}
	double d_all_entropy = 0.;
	for (auto itr = mp.begin(); itr != mp.end(); ++itr) 
	{
		d_all_entropy += (cal_vec_entropy(itr->second, f_cc) * itr->second.size() / vdata.size());
	}
	return d_all_entropy;
}

template<typename param_classifier_t, typename class_classifier_t, int dim_size>
int max_entropy_gain_index(const std::vector<mat<dim_size, 1, double> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	double d_all = cal_vec_entropy(vdata, f_cc);
	double d_max_gain = -1e10;
	int i_max_idx = 0;
	for (int i = 0; i < dim_size-1; ++i)
	{
		double d_cur_gain = d_all - cal_div_entropy(vdata, i, f_pc, f_cc);
		if (d_cur_gain > d_max_gain) 
		{
			d_max_gain = d_cur_gain;
			i_max_idx = i;
		}
	}
	return i_max_idx;
}

/* 根据idx位置的值分割数据集，返回值为idx位置类别到数据集的映射 */
template<typename param_classifier_t, typename class_classifier_t, int dim_size>
std::map<int, std::vector<mat<dim_size, 1, double> > > div_data(const std::vector<mat<dim_size, 1, double> >& vdata, const int& idx, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	std::map<int, std::vector<mat<dim_size, 1, double> > > mp;
	for (auto itr = vdata.begin(); itr != vdata.end(); ++itr)
	{
		const mat<dim_size, 1, double>& mt = *itr;
		int i_class = f_pc(idx, mt[idx]);
		if (mp.count(i_class) == 0)
		{
			mp.insert(std::make_pair(i_class, std::vector<mat<dim_size, 1, double> >()));
		}
		mp[i_class].push_back(mt);
	}
	return mp;
}

/* 判断类型是不是一类 */
template<typename class_classifier_t, int dim_size>
bool same_class(int& i_class, const std::vector<mat<dim_size, 1, double> >& vdata,  class_classifier_t& f_cc)
{
	i_class = f_cc(vdata[0][dim_size - 1]);
	if (vdata.size() < 2) 
	{
		return true;
	}
	for (int i = 1; i < vdata.size(); ++i) 
	{
		int i_cur_class = f_cc(vdata[i][dim_size - 1]);
		if (i_class != i_cur_class) 
		{
			return false;
		}
	}
	return true;
}

struct dt_node 
{
	bool is_leave;							// 是否为叶子
	int idx;								// 当前节点索引
	int lbl;								// 当前节点类别
	double rate;							// 正确率
	std::map<int, dt_node*>	mp_sub;			// 下层节点
	dt_node() :is_leave(false), lbl(-1), idx(-1), rate(1.)
	{}
};

template<typename param_classifier_t, typename class_classifier_t, int dim_size>
void _gen_id3_tree(struct dt_node* p_cur_node, const std::vector<mat<dim_size, 1, double> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	int i_class = 0;
	if (same_class(i_class, vdata, f_cc)) 
	{
		p_cur_node->lbl = i_class;
		p_cur_node->is_leave = true;
		return;
	}
	p_cur_node->idx = max_entropy_gain_index(vdata, f_pc, f_cc);														// 获取最大分割索引
	std::map<int, std::vector<mat<dim_size, 1, double> > > mp_div = div_data(vdata, p_cur_node->idx, f_pc, f_cc);		// 分割数据集
	for (auto itr = mp_div.begin(); itr != mp_div.end(); ++itr)															// 循环判断子集合的决策树
	{
		struct dt_node* p_sub_node = new struct dt_node();																// 创建一个新的节点
		_gen_id3_tree(p_sub_node, itr->second, f_pc, f_cc);																	// 生成子数据集的决策树
		p_cur_node->mp_sub.insert(std::make_pair(itr->first, p_sub_node));												// 将子决策树加到当前决策树的下面
	}
}

template<typename param_classifier_t, typename class_classifier_t, int dim_size>
dt_node* gen_id3_tree(const std::vector<mat<dim_size, 1, double> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	struct dt_node* p_tree = new struct dt_node();
	_gen_id3_tree(p_tree, vdata, f_pc, f_cc);
	return p_tree;
}

template<typename param_classifier_t, int dim_size>
int judge_id3(struct dt_node* p_cur_node, const mat<dim_size, 1, double>& data, param_classifier_t& f_pc, const int& def_value)
{
	if (p_cur_node->is_leave) 
	{
		return p_cur_node->lbl;
	}
	int i_next_idx = f_pc(p_cur_node->idx, data[p_cur_node->idx]);
	if (p_cur_node->mp_sub.count(i_next_idx) == 0)			// 之前训练时候没有遇到过的分类
	{
		return def_value;
	}
	return judge_id3(p_cur_node->mp_sub[i_next_idx], data, f_pc, def_value);
}

template<typename param_classifier_t, typename class_classifier_t, int dim_size>
std::tuple<double, double> cal_expect_entropy_and_iv(const std::vector<mat<dim_size, 1, double> >& vdata, const int& idx, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	std::map<int, std::vector<mat<dim_size, 1, double> > > mp;				// 按照idx进行分类，分类后的类别到数据的映射就是这个
	for (auto itr = vdata.begin(); itr != vdata.end(); ++itr)
	{
		const mat<dim_size, 1, double>& mt = *itr;
		int i_class = f_pc(idx, mt[idx]);
		if (mp.count(i_class) == 0)
		{
			mp.insert(std::make_pair(i_class, std::vector<mat<dim_size, 1, double> >()));
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

template<typename param_classifier_t, typename class_classifier_t, int dim_size>
int max_entropy_gain_ratio_index(const std::vector<mat<dim_size, 1, double> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	double d_all = cal_vec_entropy(vdata, f_cc);
	double d_max_gain = -1e10;
	int i_max_idx = 0;
	for (int i = 0; i < dim_size - 1; ++i)
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

template<typename param_classifier_t, typename class_classifier_t, int dim_size>
void _gen_c45_tree(struct dt_node* p_cur_node, const std::vector<mat<dim_size, 1, double> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	int i_class = 0;
	if (same_class(i_class, vdata, f_cc))
	{
		p_cur_node->lbl = i_class;
		p_cur_node->is_leave = true;
		return;
	}
	p_cur_node->idx = max_entropy_gain_ratio_index(vdata, f_pc, f_cc);														// 获取最大分割索引
	std::map<int, std::vector<mat<dim_size, 1, double> > > mp_div = div_data(vdata, p_cur_node->idx, f_pc, f_cc);		// 分割数据集
	for (auto itr = mp_div.begin(); itr != mp_div.end(); ++itr)															// 循环判断子集合的决策树
	{
		struct dt_node* p_sub_node = new struct dt_node();																// 创建一个新的节点
		_gen_c45_tree(p_sub_node, itr->second, f_pc, f_cc);																	// 生成子数据集的决策树
		p_cur_node->mp_sub.insert(std::make_pair(itr->first, p_sub_node));												// 将子决策树加到当前决策树的下面
	}
}

template<typename param_classifier_t, typename class_classifier_t, int dim_size>
dt_node* gen_c45_tree(const std::vector<mat<dim_size, 1, double> >& vdata, param_classifier_t& f_pc, class_classifier_t& f_cc)
{
	struct dt_node* p_tree = new struct dt_node();
	_gen_c45_tree(p_tree, vdata, f_pc, f_cc);
	return p_tree;
}

template<typename param_classifier_t, int dim_size>
int judge_c45(struct dt_node* p_cur_node, const mat<dim_size, 1, double>& data, param_classifier_t& f_pc, const int& def_value)
{
	if (p_cur_node->is_leave)
	{
		return p_cur_node->lbl;
	}
	int i_next_idx = f_pc(p_cur_node->idx, data[p_cur_node->idx]);
	if (p_cur_node->mp_sub.count(i_next_idx) == 0)			// 之前训练时候没有遇到过的分类
	{
		return def_value;
	}
	return judge_c45(p_cur_node->mp_sub[i_next_idx], data, f_pc, def_value);
}

#endif
