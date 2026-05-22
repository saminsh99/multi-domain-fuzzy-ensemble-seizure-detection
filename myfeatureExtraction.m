function [features] = myfeatureExtraction(sig)
     mu = mean(sig);
     v = var(sig);
     sk = skewness(sig);
     kr = kurtosis(sig);
     p = mean(sig.^2);
     med = median(sig);
     %rng = range(sig);
     st = std(sig);
     line_length = sum(abs(diff(sig))); % Line length
     % Calculate peak-to-peak amplitude
     %ptp_amplitude = max(sig) - min(sig);
     line_length = sum(abs(diff(sig))); % Line length

     features = [mu; p; med;line_length;st;kr];
end

