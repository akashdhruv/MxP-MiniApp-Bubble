// params.h -- runtime-parameter struct and .ini parsing.
//
// Spec: dev/tasks/init-project/spec.md ("Physics and numerics contract",
// "Parallel contract", "Output and post-processing").
#ifndef BUBBLE_PARAMS_H
#define BUBBLE_PARAMS_H

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

// WENO5 reaches i-3 .. i+3.  Fewer guard cells is a silent correctness bug.
constexpr int NGUARD = 3;

struct Params {
    // mesh
    int nx = 128;             // interior cells in x (global)
    int ny = 128;             // interior cells in y (global)
    double xmin = 0.0;
    double xmax = 1.0;
    double ymin = 0.0;
    double ymax = 1.0;

    // decomposition (0 => let MPI_Dims_create choose)
    int px = 0;
    int py = 0;

    // initial condition: phi = radius - |x - center|
    double bubble_x = 0.75;
    double bubble_y = 0.75;
    double bubble_r = 0.1;

    // time integration
    double tmax = 2.0;
    double cfl = 0.5;
    double fixed_dt = 0.0;    // > 0 overrides the CFL estimate
    int max_steps = 0;        // > 0 caps the step count (tests)
    bool reverse_time = false;  // multiply velocity by cos(pi t / tmax)

    // level set
    int ls_iterations = 2;    // mph_lsIt

    // output
    std::string outdir = "out";
    int plot_interval = 20;   // steps between .vti/.pvti dumps (<=0 disables)
    int hist_interval = 1;    // steps between history.csv rows
    int verbosity = 1;
};

namespace paramsdetail {

inline std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    const std::string::size_type b = s.find_first_not_of(ws);
    if (b == std::string::npos) return std::string();
    const std::string::size_type e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

inline bool toBool(const std::string& v, bool& out) {
    std::string s;
    for (char c : v) s += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "1" || s == "true" || s == "yes" || s == "on") { out = true; return true; }
    if (s == "0" || s == "false" || s == "no" || s == "off") { out = false; return true; }
    return false;
}

inline bool toInt(const std::string& v, int& out) {
    std::istringstream is(v);
    long long tmp = 0;
    is >> tmp;
    if (!is || !is.eof()) return false;
    out = static_cast<int>(tmp);
    return true;
}

inline bool toDouble(const std::string& v, double& out) {
    std::istringstream is(v);
    double tmp = 0.0;
    is >> tmp;
    if (!is || !is.eof()) return false;
    out = tmp;
    return true;
}

}  // namespace paramsdetail

// Apply one `key = value` assignment.  Returns false on an unknown key
// (`known` set to false) or an unparsable value (`known` true).
inline bool setParam(const std::string& key, const std::string& val, Params& p, bool& known) {
    known = true;
    bool ok = true;
    if (key == "nx") ok = paramsdetail::toInt(val, p.nx);
    else if (key == "ny") ok = paramsdetail::toInt(val, p.ny);
    else if (key == "xmin") ok = paramsdetail::toDouble(val, p.xmin);
    else if (key == "xmax") ok = paramsdetail::toDouble(val, p.xmax);
    else if (key == "ymin") ok = paramsdetail::toDouble(val, p.ymin);
    else if (key == "ymax") ok = paramsdetail::toDouble(val, p.ymax);
    else if (key == "px") ok = paramsdetail::toInt(val, p.px);
    else if (key == "py") ok = paramsdetail::toInt(val, p.py);
    else if (key == "bubble_x") ok = paramsdetail::toDouble(val, p.bubble_x);
    else if (key == "bubble_y") ok = paramsdetail::toDouble(val, p.bubble_y);
    else if (key == "bubble_r") ok = paramsdetail::toDouble(val, p.bubble_r);
    else if (key == "tmax") ok = paramsdetail::toDouble(val, p.tmax);
    else if (key == "cfl") ok = paramsdetail::toDouble(val, p.cfl);
    else if (key == "fixed_dt") ok = paramsdetail::toDouble(val, p.fixed_dt);
    else if (key == "max_steps") ok = paramsdetail::toInt(val, p.max_steps);
    else if (key == "reverse_time") ok = paramsdetail::toBool(val, p.reverse_time);
    else if (key == "ls_iterations") ok = paramsdetail::toInt(val, p.ls_iterations);
    else if (key == "outdir") p.outdir = val;
    else if (key == "plot_interval") ok = paramsdetail::toInt(val, p.plot_interval);
    else if (key == "hist_interval") ok = paramsdetail::toInt(val, p.hist_interval);
    else if (key == "verbosity") ok = paramsdetail::toInt(val, p.verbosity);
    else { known = false; return false; }
    return ok;
}

// Apply a command-line override of the form `key=value`.
inline bool parseParamAssignment(const std::string& arg, Params& p, std::string& err) {
    const std::string::size_type eq = arg.find('=');
    if (eq == std::string::npos) {
        err = "expected 'key=value', got '" + arg + "'";
        return false;
    }
    const std::string key = paramsdetail::trim(arg.substr(0, eq));
    const std::string val = paramsdetail::trim(arg.substr(eq + 1));
    bool known = true;
    if (!setParam(key, val, p, known)) {
        err = known ? ("bad value '" + val + "' for '" + key + "'")
                    : ("unknown parameter '" + key + "'");
        return false;
    }
    return true;
}

// Parse a flat `key = value` .ini file.  `#` and `;` start a comment,
// `[section]` headers are ignored.  Returns false and fills `err` on an
// unknown key or an unparsable value -- bad input is never silently ignored.
inline bool parseParamFile(const std::string& path, Params& p, std::string& err) {
    std::ifstream f(path.c_str());
    if (!f) {
        err = "cannot open parameter file '" + path + "'";
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        const std::string::size_type hash = line.find_first_of("#;");
        if (hash != std::string::npos) line = line.substr(0, hash);
        std::string s = paramsdetail::trim(line);
        if (s.empty()) continue;
        if (s[0] == '[') continue;
        const std::string::size_type eq = s.find('=');
        if (eq == std::string::npos) {
            err = path + ":" + std::to_string(lineno) + ": expected 'key = value'";
            return false;
        }
        const std::string key = paramsdetail::trim(s.substr(0, eq));
        const std::string val = paramsdetail::trim(s.substr(eq + 1));
        bool known = true;
        const bool ok = setParam(key, val, p, known);
        if (!known) {
            err = path + ":" + std::to_string(lineno) + ": unknown parameter '" + key + "'";
            return false;
        }
        if (!ok) {
            err = path + ":" + std::to_string(lineno) + ": bad value '" + val +
                  "' for '" + key + "'";
            return false;
        }
    }
    return true;
}

// Range / consistency checks that do not need MPI.
inline bool validateParams(const Params& p, std::string& err) {
    if (p.nx <= 0 || p.ny <= 0) { err = "nx and ny must be positive"; return false; }
    if (p.xmax <= p.xmin || p.ymax <= p.ymin) { err = "domain extents must be increasing"; return false; }
    if (p.px < 0 || p.py < 0) { err = "px and py must be non-negative"; return false; }
    if (p.tmax <= 0.0) { err = "tmax must be positive"; return false; }
    if (p.cfl <= 0.0 || p.cfl > 1.0) { err = "cfl must lie in (0,1]"; return false; }
    if (p.fixed_dt < 0.0) { err = "fixed_dt must be non-negative"; return false; }
    if (p.max_steps < 0) { err = "max_steps must be non-negative"; return false; }
    if (p.ls_iterations < 0) { err = "ls_iterations must be non-negative"; return false; }
    if (p.bubble_r <= 0.0) { err = "bubble_r must be positive"; return false; }
    if (p.outdir.empty()) { err = "outdir must not be empty"; return false; }
    return true;
}

#endif  // BUBBLE_PARAMS_H
