#ifndef MATERSDK_MTPR_MTP_PARAM_H
#define MATERSDK_MTPR_MTP_PARAM_H
#include <string>
#include <vector>
#include <set>

namespace matersdk {
namespace mtpr {

class MtpException {
public:
    std::string message;

    MtpException(const std::string& message) : message(message) {}

    virtual const char* what() const {
        return this->message.c_str();
    }
};  // class : MtpException

#define MtpError(str) throw MtpException((std::string)"ERROR: " + str + \
    " \n    Thrown by function " + __FUNCTION__ + \
    ",\n    source file " + __FILE__ + \
    ",\n    line " + std::to_string(static_cast<long long>(__LINE__)) + "\n");

class MtpParam {
public:
    MtpParam();

    MtpParam(const std::string& filename);

    void _load(const std::string& filename);

    MtpParam(const MtpParam& rhs);

    MtpParam(MtpParam&& rhs);

    MtpParam& operator=(const MtpParam& rhs);

    MtpParam& operator=(MtpParam&& rhs);

    ~MtpParam();

    std::set<int> _get_mus4all_mom(int mom_idx);

    std::vector<std::set<int>> _get_mus4all_mom_dp(int num_moms);

    void show() const;

    const int alpha_moments_count() const;

    const int alpha_index_basic_count() const;

    const int (*alpha_index_basic() const)[4];

    const int alpha_index_times_count() const;

    const int (*alpha_index_times() const)[4];

    const int alpha_scalar_moments() const;

    const int *alpha_moment_mapping() const;

    const int max_num_mus4mom() const;

    const int *num_mus4moms() const;

    const int *mus4moms_ptr() const;

    const int nmus() const;

private:
    int _alpha_moments_count = 0;
    int _alpha_index_basic_count = 0;
    int (*_alpha_index_basic)[4] = nullptr;
    int _alpha_index_times_count = 0;
    int (*_alpha_index_times)[4] = nullptr;
    int _alpha_scalar_moments = 0;
    int *_alpha_moment_mapping = nullptr;
    int _max_num_mus4mom = 0;
    int *_num_mus4moms = nullptr;
    int *_mus4moms_ptr = nullptr;
    //int _alpha_count = 0;
};  // class MtpParam


class AlphaIndexTimes {
public:
    AlphaIndexTimes();

    AlphaIndexTimes(
        int alpha_index_times_count,
        int (*alpha_index_times)[4]);
    
    AlphaIndexTimes(const AlphaIndexTimes &rhs);

    AlphaIndexTimes(AlphaIndexTimes &&rhs);

    AlphaIndexTimes& operator=(const AlphaIndexTimes &rhs);

    AlphaIndexTimes& operator=(AlphaIndexTimes &&rhs);

    ~AlphaIndexTimes();

    void show() const;

    const int alpha_index_times_count() const;

    const int (*alpha_index_times() const)[4];

private:
    int _alpha_index_times_count = 0;
    int (*_alpha_index_times)[4] = nullptr;
};  // class : AlphaIndexTimes


};  // namespace : mtpr
};  // namespace : matersdk

#endif