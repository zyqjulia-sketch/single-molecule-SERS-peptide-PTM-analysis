%% === USER PARAMETERS ===
dataFile   = 'Spectra_clean_normalizedt_EYAA4_HIF7SPTM_Mono_AuNP50nm_pH7_N1 .xlsx';          
outputFile = 'peak_occurrences_EYAA4_HIF7SPTM_Mono_AuNP50nm_pH7_N1.xlsx';   

figurename='EYAA4_HIF7SPTM_Mono_AuNP50nm_pH7_N1';

minProminence = 0.1;                   
minHeight     = 0.35;                   
minWidth      = 8;  
tolerance     = 1.5; 
normalizeType = 'max';                  

numExamplePlots = 3;                    

spectraStart=1;                         
spectraEnd=981;

%% === SMOOTHING PARAMETERS ===
applySmoothing = true;                  % true / false
smoothMethod   = 'sgolay';              % 'sgolay', 'movmean', 'gaussian'
smoothWindow   = 8;                    % Must be odd for sgolay

%% === LOAD DATA (WITH HEADERS) ===
if ~isfile(dataFile)
    error('Input file "%s" not found. Please check the path.', dataFile);
end

T = readtable(dataFile);

ramanShift = T{spectraStart:spectraEnd,1};
spectra    = T{spectraStart:spectraEnd,2:end};
spectrumNames = T.Properties.VariableNames(2:end);
numSpectra = size(spectra,2);

fprintf('Loaded %d spectra with %d Raman shift points.\n', numSpectra, numel(ramanShift));

%% === INITIALIZE COUNTER ===
peakCount = zeros(size(ramanShift));

%% === NORMALIZE + SMOOTH + FIND PEAKS ===
for i = 1:numSpectra
    y = spectra(:,i);

    % --- Optional Smoothing ---
    if applySmoothing
        y = smoothdata(y, smoothMethod, smoothWindow);
    end

    % --- Find peaks ---
    [pks, locs, w, p] = findpeaks(y, ramanShift, ...
        'MinPeakProminence', minProminence, ...
        'MinPeakHeight',     minHeight, ...
        'MinPeakWidth',      minWidth);

    fprintf('Spectrum %d (%s): %d peaks found.\n', i, spectrumNames{i}, numel(pks));

    % --- Count occurrences ---
    for j = 1:numel(locs)
        [~, idx] = min(abs(ramanShift - locs(j)));
        peakCount(idx) = peakCount(idx) + 1;
    end

    % --- Visualize a few spectra ---
    if i <= numExamplePlots
        figure('Color','w');
        plot(ramanShift, spectra(:,i), 'Color',[0.7 0.7 0.7]); hold on;
        plot(ramanShift, y, 'b-', 'LineWidth', 1.2);
        plot(locs, pks, 'ro', 'MarkerFaceColor','r');
        title(sprintf('Spectrum %d (%s) — Smoothed (%s)', ...
            i, spectrumNames{i}, smoothMethod));
        xlabel('Raman Shift (cm^{-1})', 'FontWeight','bold');
        ylabel('Intensity', 'FontWeight','bold');
        grid on; set(gca, 'FontSize', 12);
        legend('Raw','Smoothed','Detected Peaks');
    end
end

%% === CALCULATE FRACTION OF SPECTRA ===
peakFraction = peakCount / numSpectra;

%% === PLOT AGGREGATE RESULTS ===
figure('Color', 'w');
yyaxis left;
bar(ramanShift, peakCount, ...
    'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none');
ylabel('Peak Occurrence Count', 'FontWeight', 'bold');

yyaxis right;
plot(ramanShift, peakFraction, 'r-', 'LineWidth', 1.5);
ylabel('Fraction of Spectra', 'FontWeight', 'bold');

xlabel('Raman Shift (cm^{-1})', 'FontWeight', 'bold');
title(figurename);
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.2);
legend('Peak Count','Fraction of Spectra','Location','best');

%% === EXPORT RESULTS ===
T_out = table(ramanShift, peakCount, peakFraction, ...
    'VariableNames', {'RamanShift_cm1', 'PeakCount', 'FractionOfSpectra'});

try
    writetable(T_out, outputFile);
    fprintf('✅ Peak occurrence data exported to "%s".\n', outputFile);
catch ME
    warning('⚠️ Could not write to "%s": %s', outputFile, ME.message);
end