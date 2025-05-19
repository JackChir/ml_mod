/**
 * @file gmm_t.hpp
 * @brief 高斯混合模型
 * @author jingke
 * @date 2025-5-19
 * @details
 * 1. 该模型使用EM算法进行训练
 * 2. 该模型使用高斯分布进行建模
 * 3. 该模型使用最大似然估计进行参数估计
 * 4. 该模型使用K均值算法进行初始化
 * 5. 该模型使用高斯分布进行建模
 * 6. 该模型使用EM算法进行训练
 */
#ifndef _GMM_T_HPP_
#define _GMM_T_HPP_
#include <vector>
#include <functional>

#include "mat.hpp"
#include "base_function.hpp"

// 求x在均值u方差sigma下的高斯分布概率
// $$ p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^d |\Sigma|}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) $$
template<int dim_size>
double poss(const mat<dim_size, 1, double>& x, const mat<dim_size, 1, double>& u, const mat<dim_size, dim_size, double>& sigma)
{
	mat<dim_size, dim_size, double> E_1 = inverse(sigma);
	mat<dim_size, 1, double> x_u = x - u;
	double _E_1_2 = (sqrt(det(sigma))*pow(2.*3.1415926535897932384626, dim_size / 2.));
	return exp(x_u.t().dot(E_1.dot(x_u))*-0.5)[0] / _E_1_2;
}

template<int dim_size>
struct em_class
{
	std::vector<int>						vec;
	mat<dim_size, 1, double>				u;
	mat<dim_size, dim_size, double>			sigma;
	double									p;
};

// 生成一个函数，计算x在第k个高斯分布下的概率
template<int dim_size>
std::function<double(const mat<dim_size, 1, double>&)> gen_gama(const std::vector<mat<dim_size, 1, double> >& vec_x, const std::vector<em_class<dim_size> >& vec_cls, const int& k)
{
	return [&vec_x, &vec_cls, k](const mat<dim_size, 1, double>& x)
	{
		double poss_all = 0.;
		for (int j = 0; j < vec_cls.size(); ++j)
		{
			poss_all += (vec_cls[j].p * poss(x, vec_cls[j].u, vec_cls[j].sigma));
		}
		return vec_cls[k].p * poss(x, vec_cls[k].u, vec_cls[k].sigma) / poss_all;
	};
}

template<int dim_size>
bool update_class(const std::vector<mat<dim_size, 1, double> >& vec_x, std::vector<em_class<dim_size> >& vec_cls)
{
	double delta_u_max = 0.;
	double delta_sigma_max = 0.;
	double N = vec_x.size();
	std::vector<em_class<dim_size> > vec_cls_new(vec_cls.size());
	for (int k = 0; k < vec_cls.size(); ++k)
	{
		mat<dim_size, 1, double> uk = 0.;
		mat<dim_size, dim_size, double> sigmak = 0.;
		//auto gama = gen_gama(vec_x, vec_cls, k);
        std::vector<double> vec_gama(vec_x.size(), 0.);
		double Nk = 0.;
		for (int n = 0; n < vec_x.size(); ++n)
		{
            auto xn = vec_x[n];
            //auto r_nk = gama(xn); // xn对k的概率

            double poss_all = 0.;
            for (int j = 0; j < vec_cls.size(); ++j)
            {
                poss_all += (vec_cls[j].p * poss(xn, vec_cls[j].u, vec_cls[j].sigma));
            }
            auto r_nk = vec_cls[k].p * poss(xn, vec_cls[k].u, vec_cls[k].sigma) / poss_all;
            vec_gama[n] = r_nk;
            uk = uk + r_nk * xn;
            Nk += r_nk; // 固定类别k，对所有数据求\gama_{ik}的和
        }
        
		uk = uk / Nk;
		for (int n = 0; n < vec_x.size(); ++n)
		{
			auto xn = vec_x[n];
			sigmak = sigmak + vec_gama[n]/Nk * (vec_x[n] - uk).dot((vec_x[n] - uk).t());
		}
		double pk = Nk / N;

		vec_cls_new[k].u = uk;
		vec_cls_new[k].sigma = sigmak;
		vec_cls_new[k].p = pk;

		auto delta_u = (vec_cls[k].u - uk).max_abs();
		auto delta_sigma = (vec_cls[k].sigma - sigmak).max_abs();
		delta_u_max = (delta_u > delta_u_max ? delta_u : delta_u_max);
		delta_sigma_max = (delta_sigma > delta_sigma_max ? delta_sigma : delta_sigma_max);

	}
	vec_cls.swap(vec_cls_new);
	if (delta_u_max < 0.0001 && delta_sigma_max < 0.001)
		return true;
	return false;
}

template<int dim_size>
void gmm(const std::vector<mat<dim_size, 1, double> >& vec_x, std::vector<struct em_class<dim_size> >& vec_cls, const int& max_loop_num = 1000)
{
	for (int lp = 0; lp < max_loop_num; ++lp)
	{
		if (update_class(vec_x, vec_cls))
		{
			printf("%d			exit\r\n", lp);
			break;
		}
	}
}

#endif
