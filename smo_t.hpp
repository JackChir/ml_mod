#ifndef _SVM_HPP_
#define _SVM_HPP_
#include <math.h>

template<typename y_t, typename k_t>
bool match_ktt(const y_t& y, const y_t& a, const y_t& g, const typename y_t::type& C, const k_t& k)
{
	/*
	KKT条件：
		yg >= 1 s.t. a[i]==0
		yg == 1 s.t. 0<a[i]<C
		yg <= 1 s.t. a[i]==C
	*/
	typename y_t::type a_sum = 0.;
	bool ret = true;
	for (int i = 0; i < y.size(); ++i)
	{
		auto v_judge = y[i] * g[i];
		if (abs(a[i]) < 1e-8)					// a[i]==0
		{
			if (v_judge < 1)				// yg >= 1
			{
				ret = false;
				break;
			}
		}
		else if (abs(a[i] - C) < 1e-8)		// a[i]==C
		{
			if (v_judge > 1)				// yg <= 1
			{
				ret = false;
				break;
			}
		}
		else									//  0<a[i]<C
		{
			if (abs(v_judge - 1.) > 1e-8)		//  yg == 1
			{
				ret = false;
				break;
			}
		}
		a_sum += (a[i] * y[i]);
	}
	if (abs(a_sum) > 1e-8) 
	{
		ret = false;
	}
	return ret;
}

template<typename val_t>
inline val_t max(const val_t& v1, const val_t& v2)
{
	return v1 < v2 ? v2 : v1;
}

template<typename val_t>
inline val_t min(const val_t& v1, const val_t& v2)
{
	return v1 > v2 ? v2 : v1;
}

template<typename x_t, typename y_t, typename k_t>
void optimize_alpha(const int& i, const int& j, const typename y_t::type& C, y_t& a, const x_t& x, const y_t& y, y_t& E, y_t& g, typename y_t::type& b, k_t& k)
{
	typename y_t::type Kii = k(x[i], x[j]), Kij = k(x[i], x[j]), Kjj = k(x[j], x[j]);
	typename y_t::type eta = Kii + Kjj - 2 * Kij + 1e-10;
	typename y_t::type a_new_unc_j = a[j] + (y[j] * (E[i] - E[j]) / eta);
	typename y_t::type L = 0., H = 0.;
	if (y[i] != y[j])
	{
		L = max(0., a[j] - a[i]);
		H = min(C, C + a[j] - a[i]);
	}
	else
	{
		L = max(0., a[i] + a[j] - C);
		H = min(C, a[i] + a[j]);
	}
	typename y_t::type a_new_j = (a_new_unc_j < L) ? L : ((a_new_unc_j > H ? H : a_new_unc_j));
	typename y_t::type a_new_i = a[i] + y[i] * y[j] * (a[j] - a_new_j);
	// 如果0<a_new<C则b_new_i应该等于b_new_j
	typename y_t::type b_new_i = -1. * E[i] - y[i] * Kii*(a_new_i - a[i]) - y[j] * Kij*(a_new_j - a[j]) + b;
	typename y_t::type b_new_j = -1. * E[j] - y[j] * Kjj*(a_new_j - a[j]) - y[i] * Kij*(a_new_i - a[i]) + b;
	typename y_t::type b_new = ((b_new_i == b_new_j) ? b_new_i : (b_new_i + b_new_j) / 2.);
	b = b_new;
	a[i] = a_new_i;
	a[j] = a_new_j;
	E = 0.; g = 0.;
	for (int i = 0; i < x.size(); ++i)
	{
		for (int lp = 0; lp < x.size(); ++lp)
		{
			g[i] = g[i] + (a[lp] * y[lp] * k(x[lp], x[i]));
		}
		g[i] = g[i] + b;
		E[i] = g[i] - y[i];
	}
	/*
	printf("i:%d\tj:%d\n", i, j);
	printf("x:\n");
	x.print();
	printf("y:\n");
	y.print();
	printf("a:\n");
	a.print();
	printf("E:\n");
	E.print();
	printf("g:\n");
	g.print();
	printf("b:%lf\n", b);
	printf("----------\n\n\n");
	*/
}

template<typename y_t>
int outer_loop(const typename y_t::type& C, const y_t& y, const y_t& g, const y_t& a)
{
	/*
	KKT条件：
		yg >= 1 s.t. a[i]==0
		yg == 1 s.t. 0<a[i]<C
		yg <= 1 s.t. a[i]==C
	*/
	int i_ret = 0;
	typename y_t::type max_delta = 1e-10;
	for (int i = 0; i < y.size(); ++i)
	{
		auto v_judge = y[i] * g[i];
		if (abs(a[i]) < 1e-8)					// a[i]==0
		{
			if (v_judge < 1)				// yg >= 1
			{
				auto cur_delta = 1. - v_judge;
				if (cur_delta > max_delta)
				{
					i_ret = i;
					max_delta = cur_delta;
				}
			}
		}
		else if (abs(a[i] - C) < 1e-8)		// a[i]==C
		{
			if (v_judge > 1)				// yg <= 1
			{
				auto cur_delta = v_judge - 1.;
				if (cur_delta > max_delta)
				{
					i_ret = i;
					max_delta = cur_delta;
				}
			}
		}
		else									//  0<a[i]<C
		{
			if (abs(v_judge - 1.) > 1e-8)		//  yg == 1
			{
				auto cur_delta = abs(v_judge - 1.);
				if (cur_delta > max_delta)
				{
					i_ret = i;
					max_delta = cur_delta;
				}
			}
		}
	}
	return i_ret;
}

template<typename y_t>
int inner_loop(const int& i, const y_t& E)
{
	int i_ret = 0;
	typename y_t::type v_max_delta = 1e-8;
	typename y_t::type Ei = E[i];
	for (int j = 0; j < E.size(); ++j)
	{
		if (i != j)
		{
			auto cur_delta = abs(Ei - E[j]);
			if (cur_delta > v_max_delta)
			{
				v_max_delta = cur_delta;
				i_ret = j;
			}
		}
	}
	return i_ret;
}

template<typename x_t, typename y_t, typename k_t>
void smo(const x_t& x, const y_t& y, y_t& a, const int& max_cnt, const typename y_t::type& C, typename y_t::type& b, k_t& k)
{
	y_t E = 0., g = 0.;
	for (int i = 0; i < x.size(); ++i)
	{
		for (int lp = 0; lp < x.size(); ++lp)
		{
			g[i] = g[i] + (a[lp] * y[lp] * k(x[lp], x[i]));
		}
		g[i] = g[i] + b;
		E[i] = g[i] - y[i];
	}
	for (int lp = 0; lp < max_cnt; ++lp)
	{
		int i = outer_loop(C, y, g, a);
		int j = inner_loop(i, E);
		optimize_alpha(i, j, C, a, x, y, E, g, b, k);
		if (match_ktt(y, a, g, C, k))
		{
			printf("loop num:%d\n", lp);
			break; // 这里kkt条件验证成功后得出的结果不正确，这里存在问题
		}
	}
	
	printf("a:\n");
	a.print();
	printf("g:\n");
	g.print();
	printf("b:%lf\n", b);
	printf("----------\n\n\n");
	
}

template<typename x_t, typename y_t, typename k_t>
void svm_train(const x_t& x, const y_t& y, const typename y_t::type& C, const int& max_cnt, y_t& a, typename y_t::type& b, const k_t& k)
{
	smo(x, y, a, max_cnt, C, b, k);
}

template<typename x_tt, typename x_t, typename y_tt, typename y_t, typename k_t>
typename y_t::type svm_predict(const x_tt& x_new, const x_t& x, const y_t& a, const y_t& y, const y_tt& b, const k_t& k)
{
	typename y_t::type ret = 0.;
	for (int lp = 0; lp < x.size(); ++lp)
	{
		ret += (a[lp] * y[lp] * k(x_new, x[lp]));
	}
	ret += b;
	return ret > 0. ? 1. : -1;
}

#endif
