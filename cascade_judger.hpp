#ifndef _CASCADE_JUDGER_H_
#define _CASCADE_JUDGER_H_
/** 
 * @file cascade_judger.h
 * @brief 级联分类器的判断器
 * @details
 * 使用决策树结合DBN构建级联分类器，DBN分别基于1/5/10分钟RSI、MACD、KDJ、十档盘口、成交量等数据进行训练，生成结果为200维度的一个向量(涨跌10%，精确到0.1%就是200个档位)，用于指明上涨下跌的幅度；
 * 1、收集一段时间内的数据；
 * 2、使用收集到的数据对DBN进行一定次数的训练；
 * 3、使用数据对决策树进行训练；
 */

 #include "mat.hpp"
 #include "dbn_t.hpp"
 #include "decision_tree.hpp"

 template<int data_num>
 struct market_data
 {
    static const int RSI_SIZE = (data_num + data_num + data_num)*3;        // RSI6/12/24、1/5/10顺序排列
    static const int MACD_SIZE = (data_num + data_num + data_num)*2;        // DIF/DEA、1/5/10顺序排列
    static const int KDJ_SIZE = (data_num + data_num + data_num)*3;        // KDJ1/5/10顺序排列
    static const int PRICE_VOLUME_SIZE = 20 * 2 * data_num;         // 10档价格/成交量顺序交叉
    mat<RSI_SIZE, 1, double>            mt_rsi;                    // RSI
    mat<MACD_SIZE, 1, double>           mt_macd;                   // MACD
    mat<KDJ_SIZE, 1, double>            mt_kdj;                    // KDJ
    mat<PRICE_VOLUME_SIZE, 1, double>   mt_price_volume;           // 10档盘口+成交量
    int label;                                                     // 10分钟后的价格标签，仅在训练数据中有效
 };

 template<int data_num>
 int get_market_data_label(const struct market_data<data_num>& data)
 {
     return data.label;
 }



// 训练时首先对各个DBN进行训练，然后再对决策树进行训练


// 预测时由于是周期性的预测，那么我们可以随时得到10分钟内所有时间的价格，需要从中找到最高最低价格，进行买卖

// RSI+价格成交量
template<typename raw_data_type>
class proxy_dbn_rsi_pv_t
{
private:
    using dbn_type = dbn_t<double, raw_data_type::RSI_SIZE + raw_data_type::PRICE_VOLUME_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<raw_data_type>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            vec_input[idx] = join_col(data.mt_rsi, data.mt_price_volume);    // 将RSI和盘口数据拼接
        }
        m_dbn.pretrain(vec_input, i_pretrain_times);    // 预训练
        // 获得期望值，对DBN进行微调
        std::vector<ret_type> vec_expect;
        vec_expect.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            auto& mt_expect = vec_expect[idx];
            mt_expect = 0.0;
            mt_expect.get(data.label, 0) = 1.0;    // 10分钟后的价格标签
        }
        m_dbn.finetune(vec_expect, i_finetune_times);    // 微调
    }

    int predict(const raw_data_type& raw_data, double& d_poss)
    {
        input_type data = join_col(raw_data.mt_rsi, raw_data.mt_price_volume);    // 将RSI和盘口数据拼接
        auto mt_out = m_dbn.forward(data);
        // 获取最大值的位置及数值
        int i_max_idx = 0;
        double d_max = -1.0;
        for (int i = 0; i < mt_out.r; ++i)
        {
            if (d_max < mt_out.get(i, 0))
            {
                d_max = mt_out.get(i, 0);
                i_max_idx = i;
            }
        }
        d_poss = d_max;
        return i_max_idx;    // 返回最大值的位置
    }
};

// MACD+价格成交量
template<typename raw_data_type>
class proxy_dbn_macd_pv_t
{
private:
    using dbn_type = dbn_t<double, raw_data_type::MACD_SIZE + raw_data_type::PRICE_VOLUME_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<raw_data_type>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            vec_input[idx] = join_col(data.mt_macd, data.mt_price_volume);    // 将MACD和盘口数据拼接
        }
        m_dbn.pretrain(vec_input, i_pretrain_times);    // 预训练
        // 获得期望值，对DBN进行微调
        std::vector<ret_type> vec_expect;
        vec_expect.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            auto& mt_expect = vec_expect[idx];
            mt_expect = 0.0;
            mt_expect.get(data.label, 0) = 1.0;    // 10分钟后的价格标签
        }
        m_dbn.finetune(vec_expect, i_finetune_times);    // 微调
    }

    int predict(const raw_data_type& raw_data, double& d_poss)
    {
        input_type data = join_col(raw_data.mt_macd, raw_data.mt_price_volume);    // 将MACD和盘口数据拼接
        auto mt_out = m_dbn.forward(data);
        // 获取最大值的位置及数值
        int i_max_idx = 0;
        double d_max = -1.0;
        for (int i = 0; i < mt_out.r; ++i)
        {
            if (d_max < mt_out.get(i, 0))
            {
                d_max = mt_out.get(i, 0);
                i_max_idx = i;
            }
        }
        d_poss = d_max;
        return i_max_idx;    // 返回最大值的位置
    }
};
// KDJ+价格成交量
template<typename raw_data_type>
class proxy_dbn_kdj_pv_t
{
private:
    using dbn_type = dbn_t<double, raw_data_type::KDJ_SIZE + raw_data_type::PRICE_VOLUME_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<raw_data_type>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            vec_input[idx] = join_col(data.mt_kdj, data.mt_price_volume);    // 将KDJ和盘口数据拼接
        }
        m_dbn.pretrain(vec_input, i_pretrain_times);    // 预训练
        // 获得期望值，对DBN进行微调
        std::vector<ret_type> vec_expect;
        vec_expect.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            auto& mt_expect = vec_expect[idx];
            mt_expect = 0.0;
            mt_expect.get(data.label, 0) = 1.0;    // 10分钟后的价格标签
        }
        m_dbn.finetune(vec_expect, i_finetune_times);    // 微调
    }

    int predict(const raw_data_type& raw_data, double& d_poss)
    {
        input_type data = join_col(raw_data.mt_kdj, raw_data.mt_price_volume);    // 将KDJ和盘口数据拼接
        auto mt_out = m_dbn.forward(data);
        // 获取最大值的位置及数值
        int i_max_idx = 0;
        double d_max = -1.0;
        for (int i = 0; i < mt_out.r; ++i)
        {
            if (d_max < mt_out.get(i, 0))
            {
                d_max = mt_out.get(i, 0);
                i_max_idx = i;
            }
        }
        d_poss = d_max;
        return i_max_idx;    // 返回最大值的位置
    }
};

// MACD+RSI
template<typename raw_data_type>
class proxy_dbn_macd_rsi_t
{
private:
    using dbn_type = dbn_t<double, raw_data_type::MACD_SIZE + raw_data_type::RSI_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<raw_data_type>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            vec_input[idx] = join_col(data.mt_macd, data.mt_rsi);    // 将MACD和RSI数据拼接
        }
        m_dbn.pretrain(vec_input, i_pretrain_times);    // 预训练
        // 获得期望值，对DBN进行微调
        std::vector<ret_type> vec_expect;
        vec_expect.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            auto& mt_expect = vec_expect[idx];
            mt_expect = 0.0;
            mt_expect.get(data.label, 0) = 1.0;    // 10分钟后的价格标签
        }
        m_dbn.finetune(vec_expect, i_finetune_times);    // 微调
    }

    int predict(const raw_data_type& raw_data, double& d_poss)
    {
        input_type data = join_col(raw_data.mt_macd, raw_data.mt_rsi);    // 将MACD和RSI数据拼接
        auto mt_out = m_dbn.forward(data);
        // 获取最大值的位置及数值
        int i_max_idx = 0;
        double d_max = -1.0;
        for (int i = 0; i < mt_out.r; ++i)
        {
            if (d_max < mt_out.get(i, 0))
            {
                d_max = mt_out.get(i, 0);
                i_max_idx = i;
            }
        }
        d_poss = d_max;
        return i_max_idx;    // 返回最大值的位置
    }
};

// 所有指标合在一起
template<typename raw_data_type>
class proxy_dbn_all_t
{
private:
    using dbn_type = dbn_t<double, raw_data_type::RSI_SIZE + raw_data_type::MACD_SIZE + raw_data_type::KDJ_SIZE + raw_data_type::PRICE_VOLUME_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<raw_data_type>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            vec_input[idx] = join_col(data.mt_rsi, data.mt_macd, data.mt_kdj, data.mt_price_volume);    // 将RSI、MACD、KDJ和盘口数据拼接
        }
        m_dbn.pretrain(vec_input, i_pretrain_times);    // 预训练
        // 获得期望值，对DBN进行微调
        std::vector<ret_type> vec_expect;
        vec_expect.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            auto& mt_expect = vec_expect[idx];
            mt_expect = 0.0;
            mt_expect.get(data.label, 0) = 1.0;    // 10分钟后的价格标签
        }
        m_dbn.finetune(vec_expect, i_finetune_times);    // 微调
    }

    int predict(const raw_data_type& raw_data, double& d_poss)
    {
        input_type data = join_col(raw_data.mt_rsi, raw_data.mt_macd, raw_data.mt_kdj, raw_data.mt_price_volume);    // 将RSI、MACD、KDJ和盘口数据拼接
        auto mt_out = m_dbn.forward(data);
        // 获取最大值的位置及数值
        int i_max_idx = 0;
        double d_max = -1.0;
        for (int i = 0; i < mt_out.r; ++i)
        {
            if (d_max < mt_out.get(i, 0))
            {
                d_max = mt_out.get(i, 0);
                i_max_idx = i;
            }
        }
        d_poss = d_max;
        return i_max_idx;    // 返回最大值的位置
    }
};

#include "cart_t.hpp"

template<int data_num>
class cascade_judger_t
{
private:
    proxy_dbn_rsi_pv_t<market_data<data_num>> m_dbn_rsi_pv;
    proxy_dbn_macd_pv_t<market_data<data_num>> m_dbn_macd_pv;
    proxy_dbn_kdj_pv_t<market_data<data_num>> m_dbn_kdj_pv;
    proxy_dbn_macd_rsi_t<market_data<data_num>> m_dbn_macd_rsi;
    proxy_dbn_all_t<market_data<data_num>> m_dbn_all;
    dt_node* m_p_decision_tree;    // 决策树
private:
    void train_dbn(const std::vector<market_data<data_num>>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        m_dbn_rsi_pv.train(vec_data, i_pretrain_times, i_finetune_times);
        m_dbn_macd_pv.train(vec_data, i_pretrain_times, i_finetune_times);
        m_dbn_kdj_pv.train(vec_data, i_pretrain_times, i_finetune_times);
        m_dbn_macd_rsi.train(vec_data, i_pretrain_times, i_finetune_times);
        m_dbn_all.train(vec_data, i_pretrain_times, i_finetune_times);
    }

    int dbn_predict(const int& idx, const market_data<data_num>& raw_data, double& d_poss)
    {
        // 先使用RSI+盘口数据进行判断
        if (idx == 0)
            return m_dbn_rsi_pv.predict(raw_data, d_poss);

        // 再使用MACD+盘口数据进行判断
        if (idx == 1)
            return m_dbn_macd_pv.predict(raw_data, d_poss);

        // 再使用KDJ+盘口数据进行判断
        if (idx == 2)
            return m_dbn_kdj_pv.predict(raw_data, d_poss);

        // 再使用MACD+RSI进行判断
        if (idx == 3)
            return m_dbn_macd_rsi.predict(raw_data, d_poss);

        // 最后使用所有指标进行判断
        if (idx == 4)
            return m_dbn_all.predict(raw_data, d_poss);
        
        return 0;           // 这里不会执行到，除非判断期的数量设置错误
    }

    void train_dt(const std::vector<market_data<data_num>>& vec_data, const double& stop_rate = 0.7)
    {
        // 训练决策树
        using dt_node_t = dt_node;
        std::function<int(const int&, const market_data<data_num>&)> pc = [this](const int& idx, const market_data<data_num>& d) {
            double d_poss = 0.0;         // DBN判断出的概率
            return this->dbn_predict(idx, d, d_poss);
        };
        m_p_decision_tree = gen_cart_tree<5>(vec_data, pc, get_market_data_label<data_num>, stop_rate);    // 生成决策树
    }
public:

    cascade_judger_t()
        : m_p_decision_tree(nullptr)
    {
    }
    ~cascade_judger_t()
    {
        if (m_p_decision_tree != nullptr)
        {
            delete m_p_decision_tree;
            m_p_decision_tree = nullptr;
        }
    }

    void train(const std::vector<market_data<data_num>>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100, const double& stop_rate = 0.7)
    {
        train_dbn(vec_data, i_pretrain_times, i_finetune_times);
        train_dt(vec_data, stop_rate);
    }

    int predict(const market_data<data_num>& raw_data, double& d_poss)
    {
        // 使用决策树进行判断
        std::function<int(const int&, const market_data<data_num>&)> pc = [this, &d_poss](const int& idx, const market_data<data_num>& d) {
            double d_poss = 0.0;         // DBN判断出的概率
            return this->dbn_predict(idx, d, d_poss);
        };
        int i_label = 0;
        std::tuple<int, double> tp = judge_cart(m_p_decision_tree, raw_data, pc, -1);
        i_label = std::get<0>(tp);
        return i_label;
    }

};

#endif
