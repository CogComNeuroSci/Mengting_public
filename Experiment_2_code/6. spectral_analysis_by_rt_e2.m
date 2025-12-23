clear; close all; clc;
set(groot, 'defaultTextInterpreter', 'none', ...
    'defaultAxesTickLabelInterpreter', 'none', ...
    'defaultLegendInterpreter', 'none');

currentDir = fullfile(fileparts(mfilename('fullpath')),'..','Experiment_2_data','beh_data');
dataDir    = fullfile(currentDir, 'results');
if ~exist(dataDir, 'dir'), mkdir(dataDir); end

conditions   = {'same_symbols', 'same_side_diff_symbols', 'different_sides'};
pad          = 32;
srate        = 20;
nARSurr      = 1000;
carange      = 1:16;
category_map = repmat({'unknown'}, numel(carange), 1);
category_map(1:4)  = {'same_symbols'};
category_map(5:8)  = {'same_side_diff_symbols'};
category_map(9:16) = {'different_sides'};  

%% === Load cleaned per-subject data===
cleanDir  = fullfile(currentDir, 'clean_data');
cfiles    = dir(fullfile(cleanDir, 'obs_*_clean.csv'));
subjects  = sort(regexprep({cfiles.name}, '_clean\.csv$', '', 'once')); 
numSubj   = numel(subjects);

rawCombined   = cell(numSubj,1);
validSubjects = [];

for i = 1:numSubj
    fcsv = fullfile(cleanDir, [subjects{i} '_clean.csv']);
    if ~isfile(fcsv), continue; end
    T = readtable(fcsv, 'Delimiter', ';');    
    T.resptime = double(T.resptime);
    T.ISI_ms = 1200 + 50 * double(T.ISI);
    idx = double(T.instr_type) + 1;
    T.condition = category_map(idx); 
    rawCombined{i}   = T;
    validSubjects(end+1) = i;
end

%% 3. Compute condition-specific spectra using RT 
nFreq       = pad/2;
specc_dev   = nan(numSubj, numel(conditions), nFreq);
specc_amp   = nan(numSubj, numel(conditions), nFreq);
freq_vector = [];
frex = []; 

for idxSubj = validSubjects
    combined = rawCombined{idxSubj};
    if isempty(combined), continue; end

    for ci = 1:numel(conditions)
        Tc = combined(strcmp(combined.condition, conditions{ci}), :);
        if isempty(Tc), continue; end
        aggC = groupsummary(Tc, 'ISI_ms', 'mean', 'resptime');
        [dC, frex, aC] = ARSurrSpectByPs(aggC.mean_resptime, srate, nARSurr, pad);
        specc_dev(idxSubj, ci, :) = dC;
        specc_amp(idxSubj, ci, :) = aC;
    end
end
if ~exist(dataDir, 'dir'); mkdir(dataDir); end

%% 5. Band-limited statistics 
alpha = 0.05; 
q_fdr = alpha;
band_idx = 6:12;
band_idx = band_idx(band_idx>=1 & band_idx<=nFreq);
[nSubj, nCond, nFreq] = size(specc_dev);
t_dev_cond_band = nan(nCond, nFreq); 
p_dev_cond_band = nan(nCond, nFreq);   
q_dev_cond_band = nan(nCond, nFreq);   
d_dev_cond_band = nan(nCond, nFreq);   

sig_dev_unc_band = false(nCond, nFreq); 
sig_dev_fdr_band = false(nCond, nFreq);  

for ci = 1:numel(conditions)
    pvec = nan(1, nFreq);
    tvec = nan(1, nFreq);
    dvec = nan(1, nFreq);
    
    % --- one-sample t-test for each frequency ---
    for fi = band_idx
        data = squeeze(specc_dev(:, ci, fi));
        [~, pvec(fi), ~, stats] = ttest(data, 0, 'Tail','right');
        tvec(fi) = stats.tstat;
        % effect size (Cohen's d)
        m  = mean(data);
        sd = std(data);
        dvec(fi) = m / sd;  
    end

    t_dev_cond_band(ci, :) = tvec;
    p_dev_cond_band(ci, :) = pvec;
    d_dev_cond_band(ci, :) = dvec;

    % --- FDR correction ---
    p_band = pvec(band_idx);
    q_band = mafdr(p_band, 'BHFDR', true);

    q_dev_cond_band(ci, band_idx) = q_band;
    sig_dev_unc_band(ci, band_idx) = p_band < alpha;
    sig_dev_fdr_band(ci, band_idx) = q_band < q_fdr;
end

mean_dev   = squeeze(mean(specc_dev,1));
sem_dev    = squeeze(std(specc_dev,0,1)) ./ sqrt(nSubj);

%% Deviation spectrum
color_map = containers.Map();
color_map('same_symbols')           = [100,142,192] / 255;
color_map('same_side_diff_symbols') = [176,210,236] / 255;
color_map('different_sides')      = [83,58,51] / 255;
abbrevs = {'SS','SSiDS','DSi'};

figure('Position',[100,100,700,400]);
set(gcf,'defaultAxesFontSize',6);
set(gcf, 'PaperPositionMode', 'auto');

subplot(2,2,1); hold on;
for ci = 1:numel(conditions)
    col = color_map(conditions{ci});
    alphaVal = 0.3;
    sb = shadedErrorBar(frex, mean_dev(ci,:), sem_dev(ci,:), ...
        'lineProps', {'-','Color',col,'LineWidth',1});
    sb.mainLine.DSiplayName = abbrevs{ci};
    sb.patch.FaceColor      = col;
    sb.patch.FaceAlpha      = alphaVal;
    set(sb.edge, 'LineWidth',0.01, 'LineStyle',':');

    sig_unc_band = sig_dev_unc_band(ci,:);
    scatter(frex(sig_unc_band), mean_dev(ci,sig_unc_band), 20, 'o', ...
        'MarkerEdgeColor','none','MarkerFaceColor',col, ...
        'DSiplayName',[abbrevs{ci} ' t-test sig']);

    sig_fdr_band = sig_dev_fdr_band(ci,:);
    scatter(frex(sig_fdr_band), mean_dev(ci,sig_fdr_band), 40, 's', ...
        'MarkerEdgeColor','none','MarkerFaceColor',col, ...
        'DSiplayName',[abbrevs{ci} ' FDR sig']);
end
title('Deviation Spectrum by RT');
xlabel('Frequency (Hz)'); ylabel('Deviation');
ylim(gca, [-0.002,0.002]);
lg2 = legend('Location','best'); lg2.Box='off'; lg2.FontSize = 5; lg2.ItemTokenSize = [10,10];
box off; hold off;

pdfFile = fullfile(dataDir, 'sig_oscillations-rt.pdf');
exportgraphics(gcf, pdfFile, 'ContentType','vector', 'BackgroundColor','none');

%% find peak fre
peakFreq_power = nan(numel(subjects), 3);
for i = 1:numel(subjects)
    peak_freqs = nan(1,3);
    freq_bins = 6:12; %~4-8hz
    for c = 1:numel(conditions)
        v = squeeze(specc_dev(i,c,:));
        [~, j] = max(v(freq_bins));
        peak_freqs(c) = frex(freq_bins(j));
    end
    peakFreq_power(i,:) = peak_freqs;  
end

validRowAny = any(~isnan(peakFreq_power),2);
raw_plot = peakFreq_power(validRowAny,:);         
z_plot   = (raw_plot - mean(raw_plot, 2)) ./ std(raw_plot, 0, 2);
z_plot(isnan(z_plot))=0;

%% boxplots of peak fre
figure('Position',[100,100,600,400]);
set(gcf,'defaultAxesFontSize',6);
set(gcf, 'PaperPositionMode', 'auto');

jitterWidth = 0.1; markerSize = 10; boxWidth = 0.32; halfBW = boxWidth/2;
baseX = repmat(1:3,sum(validRowAny),1);
custom_palette = struct(...
    'same_symbols',           [100,142,192] / 255,...
    'same_side_diff_symbols', [176,210,236] / 255,...
    'different_sides',      [83,58,51] / 255);
colors = [custom_palette.same_symbols; custom_palette.same_side_diff_symbols; custom_palette.different_sides];

ax = subplot(2,2,1); hold(ax,'on');
hb = boxplot(raw_plot, {'SS','SSiDS','DSi'}, 'Colors',colors,'Widths',boxWidth,'Notch','off');
set(hb,'LineWidth',1);
xj = baseX + (rand(size(raw_plot,1),3)-0.5)*jitterWidth;

means_raw = nanmean(raw_plot,1);
for c=1:3
    scatter(ax,xj(:,c),raw_plot(:,c),markerSize,'MarkerEdgeColor','white','MarkerFaceColor',colors(c,:),'LineWidth', 0.1);
    plot(ax,[c-halfBW,c+halfBW],[means_raw(c) means_raw(c)],':','Color',colors(c,:),'LineWidth',1);
end
xlim(ax,[0.5,3.5]); ylim(ax,[1,10]);set(gca, 'YTick', 2:2:10);
set(ax,'FontSize',6);
ylabel(ax,'Peak Frequency (Hz)','FontSize',6);
title(ax,'Peak frequency by RT','FontSize',6);
box(ax,'off');

pdfFile = fullfile(dataDir, 'Peak_frequency-rt.pdf');
exportgraphics(gcf, pdfFile, 'ContentType','vector', 'BackgroundColor','none');

%% --- prepare for LMM  ---
subjNames = string(subjects(validRowAny))';
condNames = {'SS','SSiDS','DSi'};

T_z = array2table(z_plot, 'VariableNames', condNames);
T_z.subject = subjNames;
T_z = movevars(T_z, 'subject', 'Before', 1);

T = stack(T_z, condNames, 'NewDataVariableName','peak_z', 'IndexVariableName','condition');
T.sub_id    = cellfun(@(s) str2double(regexp(s,'\d+','match','once')), cellstr(T.subject));
T           = T(:, {'subject','sub_id','condition','peak_z'});
T.subject   = string(T.subject);
T.condition = string(T.condition);

RT_mean = nan(numel(subjects), numel(conditions));
for i = 1:numel(subjects)
    Ti = rawCombined{i,1};
    if isempty(Ti), continue; end
    for ci = 1:numel(conditions)
        m = Ti.resptime(strcmp(Ti.condition, conditions{ci}));
        if ~isempty(m), RT_mean(i,ci) = mean(m, 'omitnan'); end
    end
end

A = array2table(RT_mean(validRowAny,:), 'VariableNames', condNames);
A.subject   = string(subjects(validRowAny))';
A           = movevars(A, 'subject', 'Before', 1);
A           = stack(A, condNames, 'NewDataVariableName','RT_mean', 'IndexVariableName','condition');
A.subject   = string(A.subject);
A.condition = string(A.condition);

T = innerjoin(T, A, 'Keys', {'subject','condition'});
T = T(:, {'subject','sub_id','condition','peak_z','RT_mean'});
T = sortrows(T, 'sub_id');
outLongFile = fullfile(dataDir, 'z_peak_fre_RT_long.xlsx');
if exist(outLongFile, 'file'), delete(outLongFile); end
writetable(T, outLongFile, 'Sheet', 'z_peak_long');

