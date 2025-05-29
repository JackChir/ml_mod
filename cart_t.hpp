#ifndef _CART_T_HPP_
#define _CART_T_HPP_
#include "decision_tree.hpp"

#include <vector>
#include <algorithm>

double gini(const std::vector<double>& samples) {
	double gini = 1.0;
	double total = 0.0;
	for (double sample : samples) {
		total += sample;
	}
	for (double sample : samples) {
		double p = sample / total;
		gini -= p * p;
	}
	return gini;
}

template<int pc_size, typename param_classifier_t, typename class_classifier_t, typename param_t>
void _gen_cart_tree(dt_node* p_cur_node, const std::vector<param_t>& S, param_classifier_t& vpc, class_classifier_t& cc, const double& stop_rate=0.9)
{
    int i_cur_lbl = 0;
    double f_rate = 0.;
    if (same_class(i_cur_lbl, S, cc, f_rate, stop_rate))
    {
        p_cur_node->lbl = i_cur_lbl;
        p_cur_node->is_leave = true;
        p_cur_node->rate = f_rate;
        return;
    }
    p_cur_node->lbl = i_cur_lbl;
    p_cur_node->rate = f_rate;
    // 1.遍历所有分类器，找到最大增益的分类器
    using mt = param_t;
    int i_max_pc_idx = 0, idx = 0;
    double max_gini_gain = -1e10;
    std::map<int, std::vector<mt>> mp_max_sub; // 最大基尼系数的分类方式
    std::map<int, std::vector<mt>> mp_sub;
    for (; idx < pc_size; ++idx)
    {
        mp_sub.clear();
        // 1.1.将数据使用当前的分类器进行分类
        for (auto s : S)
        {
            int i_class = vpc(idx, s);
            if (mp_sub.count(i_class) == 0)
            {
                mp_sub.insert(std::make_pair(i_class, std::vector<mt>()));
            }
            mp_sub[i_class].push_back(s);
        }
        // 1.2.统计总体数据的gini系数以及各子类加权后的gini系数
        double d_all_cnt = S.size();
        double d_gini_e = 0.;           // 基尼系数期望
        std::map<int, double> mp_count; // 对于整个数据集，统计各类数量
        for (auto itr = mp_sub.begin(); itr != mp_sub.end(); ++itr)
        {
            std::map<int, double> mp_sub_count; // 循环内变量，统计当前子类的数量
            std::vector<mt> &vec_cur_sub = itr->second;
            for (auto cur_s : vec_cur_sub)
            {
                int i_cur_s_class = cc(cur_s); // 判断样本的目标类别
                mp_sub_count[i_cur_s_class] = (mp_sub_count.count(i_cur_s_class) == 0 ? 1 : mp_sub_count[i_cur_s_class] + 1);
                mp_count[i_cur_s_class] = (mp_count.count(i_cur_s_class) == 0 ? 1 : mp_count[i_cur_s_class] + 1);
            }
            std::vector<double> vec_sub_cnt(mp_sub_count.size());
            std::transform(mp_sub_count.begin(), mp_sub_count.end(), vec_sub_cnt.begin(), [](const std::pair<int, double> &p)
                           { return p.second; });
            double d_sub_gini = gini(vec_sub_cnt);                     // 计算子类的gini系数
            d_gini_e += (vec_cur_sub.size() / d_all_cnt * d_sub_gini); // 计算数量加权后的gini系数之和
        }
        std::vector<double> vec_cnt(mp_count.size());
        std::transform(mp_count.begin(), mp_count.end(), vec_cnt.begin(), [](const std::pair<int, double> &p)
                       { return p.second; });
        double d_gini_a = gini(vec_cnt);
        double d_gini_gain = d_gini_a - d_gini_e; // 计算gini系数的增益
        // 1.3.判断并选出最大增益的分类器
        if (max_gini_gain < d_gini_gain)
        {
            mp_max_sub = mp_sub;
            max_gini_gain = d_gini_gain;
            i_max_pc_idx = idx;
        }
    }
    p_cur_node->idx = i_max_pc_idx;
    // 2.增加一个保险，如果没有分类器的增益大于0，则直接返回。这样可以防止过拟合
    if (max_gini_gain <= 1e-10)
    {
        p_cur_node->is_leave = true;
        p_cur_node->lbl = i_cur_lbl;
        return;
    }
    for (auto itr = mp_max_sub.begin(); itr != mp_max_sub.end(); ++itr) // 循环判断子集合的决策树
    {
        struct dt_node *p_sub_node = new struct dt_node();                 // 创建一个新的节点
        _gen_cart_tree<pc_size>(p_sub_node, itr->second, vpc, cc, stop_rate);      // 生成子数据集的决策树
        p_cur_node->mp_sub.insert(std::make_pair(itr->first, p_sub_node)); // 将子决策树加到当前决策树的下面
    }
}

template<int pc_size, typename param_classifier_t, typename class_classifier_t, typename param_t>
dt_node* gen_cart_tree(const std::vector<param_t>& vdata, param_classifier_t& vpc, class_classifier_t& cc, const double& stop_rate=0.9)
{
	struct dt_node* p_tree = new struct dt_node();
	_gen_cart_tree<pc_size>(p_tree, vdata, vpc, cc, stop_rate);
	return p_tree;
}

template<typename param_classifier_t, typename param_t>
std::tuple<int, double> judge_cart(struct dt_node* p_cur_node, param_t& data, param_classifier_t& vpc, const int& def_value)
{
	if (p_cur_node->is_leave)
	{
        //printf("get label[%d] with rate[%.2lf]\r\n", p_cur_node->lbl, p_cur_node->rate);
		return std::tie(p_cur_node->lbl, p_cur_node->rate);
	}
	int i_next_idx = vpc(p_cur_node->idx, data);
	if (p_cur_node->mp_sub.count(i_next_idx) == 0)			// 之前训练时候没有遇到过的分类
	{
		return std::tuple<int,double>(def_value, 1.);
	}
    //printf("next idx=%d\r\n", i_next_idx);
	return judge_cart(p_cur_node->mp_sub[i_next_idx], data, vpc, def_value);
}

#endif
