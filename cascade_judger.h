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


 template<typename dbn_name, int input_size, int...hidden_layer_size>
 auto get_dbn_ptr()
 {
    using dbn_type = dbn_t<double, input_size, hidden_layer_size..., 200>;
    static dbn_type s_dbn;
    return &s_dbn;
 }

 template<int data_num>
 struct market_data_for_train
 {
    static const int RSI_SIZE = (data_num + data_num + data_num)*3;        // RSI6/12/24、1/5/10顺序排列
    static const int MACD_SIZE = (data_num + data_num + data_num)*2;        // DIF/DEA、1/5/10顺序排列
    static const int KDJ_SIZE = (data_num + data_num + data_num)*3;        // KDJ1/5/10顺序排列
    static const int PRICE_VOLUME_SIZE = 20 * 2 * data_num;         // 10档价格/成交量顺序交叉
    mat<RSI_SIZE, 1, double>            mt_rsi;                    // RSI
    mat<MACD_SIZE, 1, double>           mt_macd;                   // MACD
    mat<KDJ_SIZE, 1, double>            mt_kdj;                    // KDJ
    mat<PRICE_VOLUME_SIZE, 1, double>   mt_price_volume;           // 10档盘口+成交量
    int label;                                                     // 10分钟后的价格标签
 };



// 训练时首先对各个DBN进行训练，然后再对决策树进行训练


// 预测时由于是周期性的预测，那么我们可以随时得到10分钟内所有时间的价格，需要从中找到最高最低价格，进行买卖

template<int data_num>
class proxy_dbn_rsi_pv_t
{
private:
    using dbn_type = typename dbn_t<double, market_data_for_train<data_num>::RSI_SIZE + market_data_for_train<data_num>::PRICE_VOLUME_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<market_data_for_train<data_num>>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            mat<data_size, 1, double> mt_input;
            for (int i = 0; i < input_type::r; ++i)
            {
                if (i < market_data_for_train<data_num>::RSI_SIZE)
                    mt_input.get(i, 0) = data.mt_rsi.get(i, 0);
                else
                    mt_input.get(i, 0) = data.mt_price_volume.get(i - market_data_for_train<data_num>::RSI_SIZE, 0);
            }
            vec_input[idx] = mt_input;
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

    int predict(const input_type& data, double& d_poss)
    {
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

template<int data_num>
class proxy_dbn_macd_pv_t
{
private:
    using dbn_type = typename dbn_t<double, market_data_for_train<data_num>::MACD_SIZE + market_data_for_train<data_num>::PRICE_VOLUME_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<market_data_for_train<data_num>>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            mat<data_size, 1, double> mt_input;
            for (int i = 0; i < input_type::r; ++i)
            {
                if (i < market_data_for_train<data_num>::MACD_SIZE)
                    mt_input.get(i, 0) = data.mt_macd.get(i, 0);
                else
                    mt_input.get(i, 0) = data.mt_price_volume.get(i - market_data_for_train<data_num>::MACD_SIZE, 0);
            }
            vec_input[idx] = mt_input;
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

    int predict(const input_type& data, double& d_poss)
    {
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

template<int data_num>
class proxy_dbn_kdj_pv_t
{
private:
    using dbn_type = typename dbn_t<double, market_data_for_train<data_num>::KDJ_SIZE + market_data_for_train<data_num>::PRICE_VOLUME_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<market_data_for_train<data_num>>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            mat<data_size, 1, double> mt_input;
            for (int i = 0; i < input_type::r; ++i)
            {
                if (i < market_data_for_train<data_num>::KDJ_SIZE)
                    mt_input.get(i, 0) = data.mt_kdj.get(i, 0);
                else
                    mt_input.get(i, 0) = data.mt_price_volume.get(i - market_data_for_train<data_num>::KDJ_SIZE, 0);
            }
            vec_input[idx] = mt_input;
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

    int predict(const input_type& data, double& d_poss)
    {
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
template<int data_num>
class proxy_dbn_macd_rsi_t
{
private:
    using dbn_type = typename dbn_t<double, market_data_for_train<data_num>::MACD_SIZE + market_data_for_train<data_num>::RSI_SIZE, 200, 200>;
    using input_type = typename dbn_type::input_type;
    using ret_type = typename dbn_type::ret_type;
    dbn_type m_dbn;
public:
    void train(const std::vector<market_data_for_train<data_num>>& vec_data, const int& i_pretrain_times = 100, const int& i_finetune_times = 100)
    {
        std::vector<input_type> vec_input;
        vec_input.resize(vec_data.size());
        for (int idx = 0; idx < vec_data.size(); ++idx)
        {
            auto&& data = vec_data[idx];
            mat<data_size, 1, double> mt_input;
            for (int i = 0; i < input_type::r; ++i)
            {
                if (i < market_data_for_train<data_num>::MACD_SIZE)
                    mt_input.get(i, 0) = data.mt_macd.get(i, 0);
                else
                    mt_input.get(i, 0) = data.mt_rsi.get(i - market_data_for_train<data_num>::MACD_SIZE, 0);
            }
            vec_input[idx] = mt_input;
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

    int predict(const input_type& data, double& d_poss)
    {
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

template<int DBN_DATA_NUM>
class cascade_judger_t
{
private:
    using dbn_macd_pv_t = typename dbn_t<double, market_data_for_train<DBN_DATA_NUM>::MACD_SIZE + market_data_for_train<DBN_DATA_NUM>::PRICE_VOLUME_SIZE, 200, 200>;
    using dbn_kdj_pv_t = typename dbn_t<double, market_data_for_train<DBN_DATA_NUM>::KDJ_SIZE + market_data_for_train<DBN_DATA_NUM>::PRICE_VOLUME_SIZE, 200, 200>;
    using dbn_macd_rsi_t = typename dbn_t<double, market_data_for_train<DBN_DATA_NUM>::MACD_SIZE + market_data_for_train<DBN_DATA_NUM>::RSI_SIZE, 200, 200>;
    using dbn_all_t = typename dbn_t<double, market_data_for_train<DBN_DATA_NUM>::RSI_SIZE + market_data_for_train<DBN_DATA_NUM>::MACD_SIZE + market_data_for_train<DBN_DATA_NUM>::KDJ_SIZE + market_data_for_train<DBN_DATA_NUM>::PRICE_VOLUME_SIZE, 200, 200>;
public:
};

#endif
