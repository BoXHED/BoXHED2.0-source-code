#include "boxhed_prep.h"
#include <stdexcept>

#ifdef _WIN32
    #include <io.h>
#elif __linux__
    #include <unistd.h>
#endif

#include <stdlib.h>
#include <numeric> 
#include <cstdlib>
#include <cmath> 

#include <omp.h>
#include <string>
#include <exception>
#include <sstream>

#define EPSILON 1e-7

#define PARALLEL

std::stringstream err;

template <class T>
using vec_iter = typename std::vector<T>::const_iterator;


template <class T>
inline bool _approx_equal (T val1, T val2){
    return std::abs(val1 - val2) < EPSILON;
}

template <class T>
inline size_t _tquant_distance(const std::vector<T>& tquant, int tquant_i, int tquant_j, T t_min,  T t_max){

    return std::max(tquant_j-tquant_i
            -static_cast<int>(_approx_equal(tquant[std::min(tquant_i, static_cast<int>(tquant.size()-1))],t_min))
            -static_cast<int>(_approx_equal(tquant[std::min(tquant_j, static_cast<int>(tquant.size()-1))],t_max)),
            0);
}


template <class T>
inline std::pair<int,int> _get_range_sorted(const std::vector<T> &vec, const T min, const T max){//, const int start_idx=-1, const int end_idx=-1){
    
    //TODO: make it work with start_idx and end_idx
    auto start_iter = vec.begin();
    auto end_iter   = vec.end();
    
    vec_iter<T> min_idx = std::lower_bound (start_iter, end_iter, min);
    vec_iter<T> max_idx = std::lower_bound (min_idx, end_iter, max);

    return std::make_pair(min_idx-vec.begin(), max_idx-vec.begin());
}


template <class T>
class preprocessor{

    public :

        preprocessor(const T* data_, const size_t nrows_, const size_t ncols_, const bool* is_cat_, const boundary_info* bndry_info_, T* out_data_, const T* quant_arr, const size_t* quant_size_arr, const size_t num_quantiles_, const size_t t_start_idx_, const size_t t_end_idx_, const size_t delta_idx_, const size_t id_col_idx_):
            data(data_),
            nrows(nrows_),
            ncols(ncols_),
            is_cat(is_cat_),
            bndry_info(bndry_info_),
            nIDs(bndry_info->nIDs),
            out_data(out_data_),
            out_nrows(bndry_info->out_nrows),
            tquant(std::vector<T>(
                        quant_arr+t_start_idx_*num_quantiles_, 
                        quant_arr+t_start_idx_*num_quantiles_
                        +quant_size_arr[t_start_idx_])),
            quant(quant_arr),
            quant_size(quant_size_arr),
            num_quantiles(num_quantiles_),
            t_start_idx(t_start_idx_),
            t_end_idx(t_end_idx_),
            dt_idx(t_end_idx),
            delta_idx(delta_idx_),
            id_col_idx(id_col_idx_)
            {}


        ~preprocessor(){
            tquant.clear();
            delete[] temp_quantized_data;
        }


        inline void quantize_column(size_t col_idx){
            auto column_quant = std::vector<T>(
                        quant + col_idx* num_quantiles, 
                        quant + col_idx* num_quantiles
                              + quant_size[col_idx]);

            for (size_t row_idx = 0; row_idx < nrows; ++row_idx){

                T val = data[row_idx*ncols + col_idx];
                T quantized_val;

                if (is_cat[col_idx]                    ||
                    //quant_size[col_idx]<num_quantiles||
                    std::isnan(val)                    ||
                    col_idx == t_start_idx             ||
                    col_idx == t_end_idx               || 
                    col_idx == id_col_idx              || 
                    col_idx == delta_idx
                    ){

                    quantized_val = val;
                }
                else{
                    auto quant_val_iter = std::lower_bound (
                            column_quant.begin(), 
                            column_quant.end(), 
                            val);

                    quant_val_iter = max(
                            --quant_val_iter, 
                            column_quant.begin());

                    quantized_val = *quant_val_iter;
                }

                temp_quantized_data[row_idx * ncols + col_idx] = quantized_val;
            }
            
        }

        void quantize_non_time_columns(){

            temp_quantized_data  = new T [nrows*ncols]; 

            //TODO: this can be further broken down into parallel parts by having blocks of the same column
            #pragma omp parallel for schedule(static)
            for (size_t col_idx=0; col_idx < ncols; ++col_idx){
                quantize_column(col_idx);
            }
            
        }

        void preprocess(){

            quantize_non_time_columns();

            #pragma omp parallel for schedule(static)
            for (size_t idx=0; idx<nIDs; ++idx){
            
                try {
                    _preprocess_one_id(idx);
                } catch (std::invalid_argument& e){
                    err<<e.what()<<std::endl;
                    throw;
                }
            }  
        }

    private:
        inline void _preprocess_one_id (size_t idx){
            const size_t in_lb  = bndry_info->in_lbs[idx];
            const size_t in_ub  = bndry_info->in_lbs[idx+1]-1;
            const size_t out_lb = bndry_info->out_lbs[idx];
            const size_t out_ub = bndry_info->out_lbs[idx+1]-1;

            size_t out_row = out_lb;
            for (size_t row = in_lb; row<=in_ub; ++row){

                //T t_start = data[row*ncols+t_start_idx];
                //T t_end   = data[row*ncols+t_end_idx];
                T t_start = temp_quantized_data[row*ncols+t_start_idx];
                T t_end   = temp_quantized_data[row*ncols+t_end_idx];


                if (t_end<=t_start)
                {
                    std::stringstream err_str;
                    err_str << "ERROR: t_end should be > t_start in input row"<<" "<<row<<".";
                    throw std::invalid_argument(err_str.str());
                }

                if (t_start < 0)
                {
                    std::stringstream err_str;
                    err_str << "ERROR: t_start should be at least 0 in input row"<<" "<<row<<".";
                    throw std::invalid_argument(err_str.str());
                }

                //TODO: lower/upper bounds can be optimized
                auto tquant_i_j = _get_range_sorted<T>(tquant, t_start, t_end);
                int tquant_i = tquant_i_j.first;
                int tquant_j = tquant_i_j.second;

                size_t out_len = 1 + _tquant_distance<T>(tquant, tquant_i, tquant_j, t_start, t_end);
                size_t curr_row_ub  = out_row + out_len;
                
                auto tquant_iter  = std::next(tquant.begin(), tquant_i);
                //XXX: -- can get out of bounds. does not happen now probably because minimum is in tquant so it does not go past it. And when we ++ a few lines down, it works now because we have precomputed them
                if (t_start != tquant[tquant_i])
                    --tquant_iter;

                bool first_row_to_fill = true;
                for (;out_row < curr_row_ub; ++out_row){
                    for (size_t col=0; col<ncols; ++col){
                        if (col==t_start_idx || col==t_end_idx)
                            continue;
                        out_data[out_row*ncols + col] = temp_quantized_data[row*ncols+col];

                    }

                    out_data[out_row*ncols + t_start_idx] = *tquant_iter;


                    if (out_len<=1){
                        out_data[out_row*ncols + dt_idx] = t_end - t_start;
                        continue;
                    }
                   
                    // DT
                    int rows_to_fill = curr_row_ub-out_row;
                    if (rows_to_fill > 1){
                        T dt_ = 0.0;
                        if (first_row_to_fill){
                            dt_ = *(std::next(tquant_iter)) - temp_quantized_data[row*ncols + t_start_idx];

                            first_row_to_fill = false;
                        } else {
                            dt_ = *(std::next(tquant_iter)) - *tquant_iter;
                        }
                        out_data [out_row*ncols + dt_idx]    = dt_;
                        out_data [out_row*ncols + delta_idx] = static_cast<T>(0);
                    }else{
                        out_data [out_row*ncols + dt_idx]    = t_end - *tquant_iter;
                    }
                    ++tquant_iter;
                }
            }
            if (out_row-1 != out_ub){
                std::stringstream err_str;
                err_str << "ERROR: loop reached its end for ID "<<out_data[in_lb*ncols + id_col_idx]<<"."<<" Check the corresponding ID data.";
                throw std::invalid_argument(err_str.str());
            }
        }

        const T* data;
        const size_t nrows;
        const size_t ncols;
        const bool* is_cat;
        const boundary_info* bndry_info;

        const size_t nIDs;

        T* out_data;
        const size_t out_nrows;
        std::vector<T> tquant;
        const T* quant;
        const size_t* quant_size;
        const size_t num_quantiles;
        T* temp_quantized_data;

        const size_t t_start_idx;
        const size_t t_end_idx;
        const size_t dt_idx;
        const size_t delta_idx;
        const size_t id_col_idx;

};



template <class T>
class id_lb_ub_calculator{

    public:

        id_lb_ub_calculator(const T* data_, const size_t nrows_, const size_t ncols_, const size_t nIDs_, const T* quant_arr, const size_t* quant_size, const size_t num_quantiles, const size_t t_start_idx_, const size_t id_col_idx_, size_t t_end_idx_):
            data(data_),
            nrows(nrows_),
            ncols(ncols_),
            nIDs(nIDs_),
            tquant(std::vector<T>(
                        quant_arr+t_start_idx_*num_quantiles, 
                        quant_arr+t_start_idx_*num_quantiles
                        + quant_size[t_start_idx_])),
            t_start_idx(t_start_idx_),
            id_col_idx(id_col_idx_),
            t_end_idx(t_end_idx_),
            in_lbs(new size_t[nIDs+1]),
            out_lbs(new size_t[nIDs+1])
            {}


        ~id_lb_ub_calculator(){
            tquant.clear();
        }

        boundary_info* get_boundaries(){
            _get_boundaries();
            boundary_info* bndry_info = new boundary_info(nIDs, out_nrows, in_lbs, out_lbs);
            return bndry_info;
        }

    private:

         void _get_boundaries(){
            //XXX: assuming data of subjects in chronological order, and contiguous

            int last_id   = data[id_col_idx], curr_id = last_id;
            size_t in_lb  = 0;
            size_t out_lb = 0;

            size_t idx = 0;
            for (size_t row=0; row<nrows; ++row){
                curr_id = data[row*ncols+id_col_idx];

                if (curr_id == last_id)
                    continue;

                size_t out_len = _out_len(in_lb, row-1);

                in_lbs[idx]=in_lb;
                out_lbs[idx]=out_lb;

                in_lb = row;
                out_lb += out_len;
                last_id = curr_id;
                idx ++;
            }

            size_t out_len = _out_len(in_lb, nrows-1);
            size_t last_ub = out_lb + out_len;

            in_lbs[idx]=in_lb;
            out_lbs [idx]=out_lb;

            in_lbs[idx+1]=nrows;
            out_lbs[idx+1]=last_ub;

            out_nrows  = last_ub;

            if (idx+1!=nIDs){
                std::stringstream err_str;
                err_str << "ERROR: The data for each ID need to be in subsequent rows.";
                throw std::invalid_argument(err_str.str());
            }
        }  

        inline size_t _out_len(size_t lb, size_t ub){
            size_t extra_len = 0;
            //TODO: this loop can still be optimized. tpart_i can be better approximated by the tpart_j of the previous iteration, assuming the intervals to be non-overlapping
            for (size_t in_row = lb; in_row <= ub; ++in_row){
                T t_start = data[in_row*ncols+t_start_idx];
                T t_end   = data[in_row*ncols+t_end_idx];

                auto tquant_i_j = _get_range_sorted<T>(tquant, t_start, t_end);
                int tquant_i = tquant_i_j.first;
                int tquant_j = tquant_i_j.second;

                extra_len += _tquant_distance<T>(tquant, tquant_i, tquant_j, t_start, t_end);
            }
            
            return (ub-lb+1) + extra_len;
        }


        const T* data;
        const size_t nrows;
        const size_t ncols;
        const size_t nIDs;
        std::vector<T> tquant;

        size_t t_start_idx;
        size_t id_col_idx;
        size_t t_end_idx;   

        size_t out_nrows;

        size_t* in_lbs;
        size_t* out_lbs;

};



template <class T>
inline void _rmv_dupl_srtd(T* arr, const size_t arr_size, size_t * out_size){

    size_t idx = 0;
    T last_val = arr[0];
    for (size_t i=1; i<arr_size; ++i){
        if (std::isnan(arr[i]))
        {
            break;
        }
        if (arr[i]!=last_val){
            arr[++idx] = arr[i];
            last_val   = arr[i];
        }
    }
    /*
    *out_size = idx+1; 
    */
    *out_size =  std::isnan(arr[idx]) ? idx : idx+1;
}


template <class T>
inline void _rmv_dupl_srtd(const std::vector<std::pair<T, size_t>> &vals, T* out, size_t * out_size){

    size_t idx = 0;
    T last_val = vals[0].first;
    out [0] = last_val; 
 
    for (size_t i=1; i<vals.size(); ++i){
        if (std::isnan(vals[i].first)){
            break;
        }
        if (vals[i].first!=last_val){
            out[++idx] = vals[i].first;
            last_val   = vals[i].first;
        }
    }
    ///*
    *out_size = idx+1; 
    //*/
    /*
    std::cout << "size:" << idx+1 << std::endl;
    throw 20;
    *out_size =  std::isnan(out[idx]) ? idx : idx+1;
    */
}


template <class T>
inline void _copy_col2arr(const T* src, size_t nrows, size_t ncols,
                      size_t col_idx, T* dst){
    for (size_t row_idx = 0; row_idx < nrows; ++row_idx){
        dst[row_idx] = src[row_idx*ncols+col_idx];
    }
}


template <class T>
inline void _rmv_nans(T *arr, size_t size, size_t *out_size){
    size_t idx = 0;
    for (size_t j = 0; j < size; ++j)
    {
        if (!std::isnan(arr[j]))
        {
            arr[idx++] = arr[j];
        }
    }
    *out_size = idx;
}

template <class T>
inline T _nan_min(T *arr, size_t size){
    T min = std::numeric_limits<T>::infinity();
    for (size_t i = 0; i < size; ++i){
        if ((!std::isnan(arr[i])) && (arr[i]<min))
            min = arr[i];
    }
    return min;
}


template <class T>
inline void _compute_quant(const T* data, size_t nrows, size_t ncols, const bool* is_cat, size_t t_start_idx, size_t t_end_idx, size_t id_idx, size_t delta_idx, T* quant, size_t* quant_size, size_t num_quantiles){
 
    #pragma omp parallel for schedule(dynamic)
    for (size_t col_idx = 0; col_idx<ncols; ++col_idx){
        if (is_cat[col_idx] || col_idx == t_end_idx || col_idx == id_idx || col_idx == delta_idx){
            continue;
        }
        size_t vals_size = (col_idx==t_start_idx) ? 2*nrows : nrows;
        //vals_size       += 1;

        /*
        T vals [vals_size];
        */
        T *vals = new T[vals_size];

        _copy_col2arr(data, nrows, ncols, col_idx, vals);
        if (col_idx == t_start_idx){
            _copy_col2arr(data, nrows, ncols, t_end_idx, vals + nrows);
        }

        //vals[vals_size-1] = (col_idx==t_start_idx) ? 0 : _nan_min(vals, vals_size-1)-1;

        size_t num_non_nan;
        _rmv_nans(vals, vals_size, &num_non_nan);
        std::stable_sort(vals, vals + num_non_nan);

        size_t num_unique;
        _rmv_dupl_srtd<T>(vals, num_non_nan, &num_unique);

        size_t num_quants = std::min(num_unique, num_quantiles-1);
        quant_size [col_idx] = num_quants;

        size_t offset;
        if (((col_idx==t_start_idx)) && (vals[0]==0)){
            offset                       = 0;
        } else {
            quant[col_idx*num_quantiles] = (col_idx==t_start_idx) ? 0 : _nan_min(vals, vals_size-1)-1;
            offset                       = 1;
            quant_size [col_idx]        += 1;
        }
        

        for (size_t i=0; i<num_quants; ++i){
            quant[col_idx*num_quantiles+i+offset] = vals[static_cast<int>(num_unique*i/num_quants)];
        }
        delete [] vals;
                
    }

}

template <class T>
inline void _fill_time_hist(const T* unique_arr, const size_t unique_arr_size, 
                            const T* data, size_t nrows, size_t ncols,
                            size_t t_start_idx, size_t t_end_idx,
                            size_t* hist){
    //TODO: maybe id_idx for searching optimization??

    std::vector<T> unique_arr_vec (unique_arr, unique_arr + unique_arr_size);

    #pragma omp parallel for schedule(dynamic) 
    for (size_t i=0; i < nrows; ++i){
        const auto t_start = data [i*ncols + t_start_idx];
        const auto t_end   = data [i*ncols + t_end_idx];
        //TODO: should I make sure they are not the same?
        auto iter_from     = std::lower_bound(unique_arr_vec.begin(), unique_arr_vec.end(), t_start);
        size_t idx_from    = static_cast<size_t>(iter_from - unique_arr_vec.begin());
 
        for (size_t i = idx_from; ; ++i){
            if (_approx_equal(unique_arr_vec[i],t_end)){
                break;
            }
            
            #pragma omp atomic
            hist [i] += 1;
        }
    }
}


template <class T>
inline void _fill_non_time_acc_weight(const std::vector<std::pair<T, size_t>> &srtd_val_idx, const T* unique, size_t num_unique,
                                       const T* data, size_t nrows, size_t ncols,
                                       size_t t_start_idx, size_t t_end_idx,// size_t col_idx,
                                       T* acc_weight){

    const size_t first_data_idx = srtd_val_idx[0].second;
    const T      first_val      = srtd_val_idx[0].first;
    const T      first_dt       = data [first_data_idx*ncols + t_end_idx] - data [first_data_idx*ncols + t_start_idx];

    acc_weight [0] = 0;
    acc_weight [1] = first_dt;
    size_t val_idx = 1;
    T last_val = first_val;

    for (size_t i = 1; i < nrows; ++i){
        const size_t data_idx = srtd_val_idx[i].second;
        const T      val      = srtd_val_idx[i].first;
        const T      dt       = data [data_idx*ncols + t_end_idx] - data [data_idx*ncols + t_start_idx];
        
        if (std::isnan(val) || val == unique[num_unique-1])
            break;
        if (val != last_val){
            ++val_idx;
            last_val = val;
            acc_weight[val_idx] = acc_weight[val_idx-1] + dt;
        }
        else{
            acc_weight [val_idx] += dt;
        }
    }
}


template <class T>
void _fill_quants (T* quant, size_t num_quantiles, size_t num_quants, size_t col_idx,
                          T* unique, T* acc_weight, size_t num_unique){
 
    std::vector<T> *acc_weight_vec = new std::vector<T>;
        
    acc_weight_vec -> resize(num_unique);

    for (size_t i = 0; i<num_unique; ++i){
        (*acc_weight_vec)[i] = acc_weight[i];
    } 

    if (num_unique <= num_quantiles){
        for (size_t i = 0; i < num_unique; ++i){
            quant[col_idx*num_quantiles + i] = unique [i];
        }
        return;
    }

    //TODO: loop can be optimized by providing lower bound
    for (size_t i = 0; i<num_quants; ++i){
        const T quant_to_select = static_cast<T>(i)/num_quants;
 
        auto iter  = std::lower_bound(acc_weight_vec -> begin(), acc_weight_vec -> end(), quant_to_select);
        size_t idx = static_cast<size_t>(max(--iter, acc_weight_vec->begin()) - acc_weight_vec->begin());
        
       
        T val   = unique[idx];
        T val_n = (idx < num_unique-1) ? unique[idx+1] : 
                                         unique[idx]+1;
        
        T w     = (*acc_weight_vec)[idx];
        T w_n   = (idx < num_unique-1) ? (*acc_weight_vec)[idx+1] : 
                                         (*acc_weight_vec)[idx]+1;
        
        T q;

        if (_approx_equal(w, quant_to_select)){
            q = val;
        } else if (_approx_equal(w_n, quant_to_select)){
            q = val_n;
        } else {
            q = ((val_n-val)/(w_n-w))*(quant_to_select-w)+val;
        }

        quant[col_idx*num_quantiles + i] = q;

        if ((i>0) && (quant[col_idx*num_quantiles + i] == quant[col_idx*num_quantiles + i - 1]))
            {
            std::stringstream err_str;
            err_str << "ERROR: An error has occured in column"<<" "<<col_idx<<" while extracting quantiles.";
            throw std::invalid_argument(err_str.str());
            }
    }
    acc_weight_vec -> clear();
}


//TODO: many of these loops can be optimized using OMP
template <class T>
inline T _compute_total_t(const T* data, size_t nrows, size_t ncols, size_t t_start_idx, size_t t_end_idx){
    
    T total_t = 0;
    for (size_t i =0; i<nrows; ++i){
        total_t += data[i*ncols + t_end_idx] - data[i*ncols + t_start_idx];
    }
    return total_t;
}


template <class T>
inline void normalize (T* arr, size_t size, T norm_factor){
    for (size_t i = 0; i<size; ++i)
        arr[i] = arr[i]/norm_factor;
}

template <class T>
void _compute_quant_weighted(const T* data, size_t nrows, size_t ncols, const bool* is_cat, size_t t_start_idx, size_t t_end_idx, size_t id_idx, size_t delta_idx, T* quant, size_t* quant_size, size_t num_quantiles){

    T total_t = _compute_total_t(data, nrows, ncols, t_start_idx, t_end_idx);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t col_idx = 0; col_idx<ncols; ++col_idx){
        if (is_cat[col_idx] || col_idx == t_end_idx || col_idx == id_idx || col_idx == delta_idx){
            continue;
        }
        size_t vals_size = (col_idx==t_start_idx) ? 2*nrows : nrows;
        vals_size       += 1;
        T *vals = new T[vals_size];

        std::vector<std::pair<T, size_t>> srtd_val_idx (vals_size);

        _copy_col2arr(data, nrows, ncols, col_idx, vals);
        if (col_idx == t_start_idx){
            _copy_col2arr(data, nrows, ncols, t_end_idx, vals + nrows);
        }

        vals[vals_size-1] = (col_idx==t_start_idx) ? 0 : _nan_min(vals, vals_size-1)-1;

        for (size_t i=0; i<vals_size; ++i){
            srtd_val_idx [i] = std::make_pair(vals[i], i);
        }

        delete [] vals;

        std::sort(srtd_val_idx.begin(), srtd_val_idx.end(), 
               [](const std::pair<T, size_t> a, 
                  const std::pair<T, size_t> b)
                 { 
                 if (std::isnan(a.first) && std::isnan(b.first)) return false; 
                   return std::isnan(b.first) || a.first < b.first; }
                 );
        
        T *unique = new T [vals_size];
        size_t num_unique;
        _rmv_dupl_srtd(srtd_val_idx, unique, &num_unique);
        
        size_t num_quants = std::min(num_unique, num_quantiles);
        quant_size [col_idx] = num_quants;

        T *acc_weight = new T[num_unique];

        if (col_idx == t_start_idx){ 
            size_t *vals_hist = new size_t [num_unique];
            std::fill_n(vals_hist, num_unique, 0);

            _fill_time_hist(unique, num_unique, 
                            data, nrows, ncols,
                            t_start_idx, t_end_idx,
                            vals_hist);
            T *time_diff = new T[num_unique];

            std::adjacent_difference (unique, unique+num_unique, time_diff);

            acc_weight[0] = 0;  
            for (size_t i = 1; i<num_unique; ++i){ //multiplying hist count by the duration
                acc_weight [i] = acc_weight[i-1] + time_diff[i] * vals_hist [i-1];
            }
                        
            delete [] time_diff;
            delete [] vals_hist;
        }
        else{
            _fill_non_time_acc_weight(srtd_val_idx, unique, num_unique, 
                            data, nrows, ncols,
                            t_start_idx, t_end_idx,// col_idx,
                            acc_weight);
            }

        normalize (acc_weight, num_unique, total_t);

        _fill_quants (quant, num_quantiles, num_quants, col_idx,
                          unique, acc_weight, num_unique);        
        delete [] unique;
        delete [] acc_weight;
        srtd_val_idx.clear();
    }
}


void free_boundary_info(boundary_info* bndry_info){
    delete bndry_info; 
}



void preprocess(
                    const void*          data_v, 
                    const size_t         nrows, 
                    const size_t         ncols, 
                    const void*          is_cat_v,
                    const boundary_info* bndry_info, 
                          void*          out_data_v, 
                    const void*          quant_v, 
                    const void*          quant_size_v,
                    const size_t         num_quantiles, 
                    const size_t         t_start_idx, 
                    const size_t         t_end_idx, 
                    const size_t         delta_idx, 
                    const size_t         id_col_idx, 
                    const int            nthreads
                    ){

    const double* data        = (double *) data_v;
    double* out_data          = (double *) out_data_v;
    bool* is_cat              = (bool   *) is_cat_v;
    const double* quant_arr   = (double *) quant_v;
    const size_t* quant_size  = (size_t *) quant_size_v;

#if defined(_OPENMP)
        omp_set_num_threads(nthreads);
#endif

    preprocessor<double> preprocessor_ (data, nrows, ncols, is_cat, bndry_info, out_data, quant_arr, quant_size, num_quantiles, t_start_idx, t_end_idx, delta_idx, id_col_idx);
    
    try {
        preprocessor_.preprocess();
    } catch (std::invalid_argument& e){
        std::cout<<err.str();
        //throw;
    }

}


boundary_info* get_boundaries(
        const void* data_v, 
        size_t nrows, 
        size_t ncols, 
        size_t nIDs, 
        size_t id_col_idx, 
        size_t t_start_idx, 
        size_t t_end_idx, 
        const void* quant_v, 
        const void* quant_size_v, 
        size_t num_quantiles
        ){

    const double* data       = (double *) data_v;
    const double* quant_arr  = (double *) quant_v;
    const size_t* quant_size = (size_t *) quant_size_v;
       
    id_lb_ub_calculator<double> id_lb_ub_calculator_ = id_lb_ub_calculator<double>(data, nrows, ncols, nIDs, quant_arr, quant_size, num_quantiles, t_start_idx, id_col_idx, t_end_idx);

    return id_lb_ub_calculator_.get_boundaries();

}



void compute_quant(
        const void* data_v, 
        size_t nrows, 
        size_t ncols, 
        void* is_cat_v, 
        size_t t_start_idx, 
        size_t t_end_idx, 
        size_t id_idx, 
        size_t delta_idx, 
        void* quant_v, 
        void* quant_size_v, 
        size_t num_quantiles, 
        bool weighted, 
        int nthreads
        ){

    const double* data = (double *) data_v;
    const bool* is_cat = (bool *)   is_cat_v;
    double* quant      = (double *) quant_v;
    size_t* quant_size = (size_t *) quant_size_v;

#if defined(_OPENMP)
        omp_set_num_threads(nthreads);
#endif

    if (weighted){
        _compute_quant_weighted<double> (data, nrows, ncols, is_cat, t_start_idx, t_end_idx, id_idx, delta_idx, quant, quant_size, num_quantiles);
    }
    else{
        _compute_quant<double> (data, nrows, ncols, is_cat, t_start_idx, t_end_idx, id_idx, delta_idx, quant, quant_size, num_quantiles);
    }
}

void shift_left(
        void* data_v, 
        size_t nrows, 
        size_t ncols, 
        const void* quant_idx_v, 
        const void* quant_v, 
        const void* quant_size_v, 
        size_t num_quantiles, 
        int nthreads
        ){

    double* data             = (double *) data_v;
    const double* quant      = (double *) quant_v;
    const size_t* quant_size = (size_t *) quant_size_v;
    const int* quant_idx     = (int *)    quant_idx_v;

    typedef double T;

#if defined(_OPENMP)
        omp_set_num_threads(nthreads);
#endif

    #pragma omp parallel for schedule(static)
    for (size_t col_idx = 0; col_idx < ncols; ++col_idx){
        size_t quant_idx_  = quant_idx  [col_idx];
        size_t quant_size_ = quant_size [quant_idx_];
        
        if (quant_size_ == 0)
            continue;

        auto column_quant = std::vector<T>(
            quant + (quant_idx_ * num_quantiles), 
            quant + (quant_idx_ * num_quantiles + quant_size_));
        
        for (size_t row_idx = 0; row_idx < nrows; ++row_idx){
            T val = data [row_idx*ncols + col_idx];

            auto quant_val_iter = std::min (std::lower_bound (
                    column_quant.begin(), 
                    column_quant.end(), 
                    val),//,
                    //[](const T a, const T b){return (a < b);}),// && (!_approx_equal(a, b));}), 
                    column_quant.end()-1);

            if (_approx_equal(val, *quant_val_iter)){
                quant_val_iter = std::max(
                    --quant_val_iter, 
                    column_quant.begin());
                data [row_idx*ncols + col_idx] = *quant_val_iter;
            }            
        }
    }
}

