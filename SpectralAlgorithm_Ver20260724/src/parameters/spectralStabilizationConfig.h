#ifndef SPECTRAL_STABILIZATION_CONFIG_H
#define SPECTRAL_STABILIZATION_CONFIG_H

struct TwoThirdsDealiasingConfig
{
    bool enabled = false;
    bool use_L2 = false; // false = rectangular, true = L2
    double cutoff_fraction = 2.0 / 3.0; // default cutoff fraction for 2/3 de-aliasing
};

struct EarlyDealiasingConfig
{
    bool enabled = false;
    bool use_L2 = false; // false = rectangular, true = L2
    double cutoff_fraction = 2.0 / 3.0; // default cutoff fraction for early de-aliasing
};

struct ZeroImaginaryPartConfig
{
    bool enabled = false;
};

struct ExponentialFilterConfig
{
    bool enabled = false;  // bật / tắt filter
    double strength = 6.0; // alpha
    double order = 8.0;    // p
    double kmax_cofficient = sqrt(2.0); // coefficient to determine kmax based on grid size (e.g., kmax = kmax_coefficient * N)
};

struct SpectralStabilizationConfig
{
    EarlyDealiasingConfig early_dealiasing = {};
    ZeroImaginaryPartConfig zero_imaginary = {};
    TwoThirdsDealiasingConfig two_thirds_dealiasing = {};
    ExponentialFilterConfig exponential_filter = {};

    void validate() const
    {
        // Exponential filter parameters
        if (exponential_filter.enabled)
        {
            if (exponential_filter.strength <= 0.0)
            {
                throw std::runtime_error(
                    "ExponentialFilterConfig: strength must be positive");
            }
            if (exponential_filter.order <= 0.0)
            {
                throw std::runtime_error(
                    "ExponentialFilterConfig: order must be positive");
            }
        }
    }
    void write_spectral_stabilization_config(const std::string &filename) const
    {
        std::ofstream outfile(filename, std::ios::out | std::ios::app);
        if (!outfile.is_open())
        {
            throw std::runtime_error(
                "Failed to open " + filename + " for writing.");
        }

        outfile.precision(constants::PRECISION);

        outfile << std::endl;
        outfile << "\n#####################################################################\n";
        outfile << "# Spectral Stabilization Config\n";
        outfile << "#####################################################################\n";

        outfile << "\nEarly de-aliasing:: "
                << (early_dealiasing.enabled ? "true" : "false") << std::endl;

        if (early_dealiasing.enabled)
        {
            outfile << "  Geometry for early de-aliasing:: "
                    << (early_dealiasing.use_L2 ? "L2 norm" : "rectangular")
                    << std::endl;
        }

        outfile << "\nZero imaginary part of the solution in Fourier space:: "
                << (zero_imaginary.enabled ? "true" : "false") << std::endl;

        outfile << "\nDe-aliasing by 2/3 rule:: "
                << (two_thirds_dealiasing.enabled ? "true" : "false") << std::endl;

        if (two_thirds_dealiasing.enabled)
        {
            outfile << "  Geometry for 2/3 de-aliasing:: "
                    << (two_thirds_dealiasing.use_L2 ? "L2 norm" : "rectangular")
                    << std::endl;
        }

        outfile << "\nExponential filter:: "
                << (exponential_filter.enabled ? "true" : "false") << std::endl;

        if (exponential_filter.enabled)
        {
            outfile << "  Exponential filter strength (alpha):: "
                    << exponential_filter.strength << std::endl;

            outfile << "  Exponential filter order (p):: "
                    << exponential_filter.order << std::endl;
                    
            outfile << "  Exponential filter kmax coefficient:: "
                    << exponential_filter.kmax_cofficient << std::endl;
        }

        outfile << "\n#####################################################################\n";

        outfile.close();
    }
};

#endif // SPECTRAL_STABILIZATION_CONFIG_H
