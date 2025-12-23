clear; close all; clc;
set(groot, 'defaultTextInterpreter','none', ...
    'defaultAxesTickLabelInterpreter','none', ...
    'defaultLegendInterpreter','none');
rootDir  = fullfile(fileparts(mfilename('fullpath')),'..','Experiment 1 data');
cleanDir = fullfile(rootDir,'beh_clean_data');
outDir   = fullfile(rootDir,'results');

if ~exist(cleanDir,'dir'); mkdir(cleanDir); end
if ~exist(outDir,'dir');   mkdir(outDir);   end

srate   = 20;     
nARSurr = 1000;   
pad     = 32;     

raw_map = containers.Map('KeyType','int32','ValueType','char');
raw_map(0) = 'LL';  % target, hand
raw_map(1) = 'RL'; 
raw_map(2) = 'LR';  
raw_map(3) = 'RR'; 

conditions = {'RR','LL','LR','RL'};

color_map = containers.Map();
color_map('RR') = [253,184,99]/255;
color_map('LL') = [230,97,1]/255;
color_map('LR') = [178,171,210]/255;
color_map('RL') = [94,60,153]/255;

abbrevs = containers.Map();
abbrevs('RR')='RR'; abbrevs('LL')='LL'; abbrevs('LR')='LR'; abbrevs('RL')='RL';

%% 1) Load cleaned data per subject
files = dir(fullfile(cleanDir, 'obs_*_beh_clean.csv'));
subjects = sort(regexprep({files.name}, '_beh_clean\.csv$', '', 'once'));
allData = cell(numel(subjects), 1);

for i = 1:numel(subjects)
    fcsv = fullfile(cleanDir, [subjects{i} '_beh_clean.csv']);
    if ~exist(fcsv, 'file'); continue; end
    T = readtable(fcsv, 'Delimiter',';');
    T.resptime = double(T.resptime);
    T.ISI_ms = 1700 + 50 * double(T.ISD);
    it = double(T.instr_type);
    cond = strings(height(T),1);
    for r = 1:height(T)
        cond(r) = string(raw_map(int32(it(r))));
    end
    T.condition = cellstr(cond);
    allData{i,1} = T;
end

%% 2) Compute spectra per subject/condition using RT
nSubj = numel(subjects);
specc_dev = nan(nSubj, numel(conditions), pad/2);
specc_amp = nan(nSubj, numel(conditions), pad/2);

for i = 1:nSubj
    Ti = allData{i,1};
    if isempty(Ti); continue; end
    
    for ci = 1:numel(conditions)
        cname = conditions{ci};
        Tc = Ti(strcmp(Ti.condition, cname), :);
        if isempty(Tc); continue; end
        aggC = groupsummary(Tc, 'ISI_ms', 'mean', 'resptime');
        [dC, frex, aC] = ARSurrSpectByPs(aggC.mean_resptime, srate, nARSurr, pad);
        specc_dev(i,ci,:) = dC;
        specc_amp(i,ci,:) = aC;
    end
end

%% 3) stats 
[~, nCond, nFreq] = size(specc_dev);
alpha = 0.05; q_fdr = alpha;

p_dev = nan(nCond, nFreq);
t_dev = nan(nCond, nFreq);
d_dev = nan(nCond, nFreq);

for ci = 1:nCond
    for fi = 1:nFreq
        data = squeeze(specc_dev(:, ci, fi));
        [~, p_dev(ci,fi), ~, stats] = ttest(data, 0, 'Tail','right');
        t_dev(ci,fi) = stats.tstat;
        m = mean(data); sd = std(data);
        d_dev(ci,fi) = m ./ sd; 
    end
end

q_dev  = nan(nCond, nFreq);
sig_fdr = false(nCond, nFreq);
for ci = 1:nCond
    q_dev(ci,:) = mafdr(p_dev(ci,:), 'BHFDR', true);
    sig_fdr(ci,:) = q_dev(ci,:) < q_fdr;
end

% limited markers
band_idx = 6:12;%~4-8hz
sig_unc_band = false(nCond, nFreq);
sig_fdr_band = false(nCond, nFreq);
for ci = 1:nCond
    pb = p_dev(ci, band_idx);
    qb = mafdr(pb, 'BHFDR', true);
    qb_condi(ci, :) = qb;
    sig_unc_band(ci, band_idx) = pb < alpha;
    sig_fdr_band(ci, band_idx) = qb < q_fdr;
end

%% 4) Plot
mean_dev = squeeze(mean(specc_dev,1));
sem_dev  = squeeze(std(specc_dev,0,1)) ./ max(1, sqrt(nSubj));

figure('Position',[120,120,720,420]);
set(gcf,'defaultAxesFontSize',7);
set(gcf, 'PaperPositionMode','auto');

subplot(2,2,1); hold on;
for ci = 1:nCond
    cname = conditions{ci};
    col = color_map(cname);
    
    sb = shadedErrorBar(frex, mean_dev(ci,:), sem_dev(ci,:), ...
        'lineProps', {'-','Color',col,'LineWidth',1});
    sb.mainLine.DisplayName = abbrevs(cname);
    sb.patch.FaceColor = col; sb.patch.FaceAlpha = 0.15;
    set(sb.edge, 'LineWidth',0.01, 'LineStyle',':');
    
    mdev = mean_dev(ci,:);
    scatter(frex(sig_unc_band(ci,:)), mdev(sig_unc_band(ci,:)), 20, 'o', ...
        'MarkerEdgeColor','none','MarkerFaceColor',col, ...
        'HandleVisibility','off');
    scatter(frex(sig_fdr_band(ci,:)), mdev(sig_fdr_band(ci,:)), 40, 's', ...
        'MarkerEdgeColor','none','MarkerFaceColor',col, ...
        'HandleVisibility','off');
end

xlabel('Frequency (Hz)'); ylabel('Deviation');
title('Deviation Spectrum by RT');
box off;
lg = legend('Location','southwest'); lg.Box='off'; lg.ItemTokenSize = [15,15]; set(lg,'AutoUpdate','off');
hold off;
ylim(gca, [-0.002,0.002]);
exportgraphics(gcf, fullfile(outDir,'sig_oscillations-rt_e1.pdf'),'ContentType','vector', 'BackgroundColor','none');

%% find peak fre
peakFreq_power = nan(numel(subjects), 4);
for i = 1:numel(subjects)
    peak_freqs = nan(1,4);
    freq_bins = 6:12; %~4-8hz
    for c = 1:nCond
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
figure('Position',[100,100,700,400]);
set(gcf,'defaultAxesFontSize',6);
set(gcf, 'PaperPositionMode', 'auto');

jitterWidth = 0.1; markerSize = 10; boxWidth = 0.4; halfBW = boxWidth/2;
baseX = repmat(1:4,sum(validRowAny),1);
col_RR = [253,184,99]/255; 
col_LL = [230,97,1]/255; 
col_LR = [178,171,210]/255; 
col_RL = [94,60,153]/255; 
box_colors = [col_RR; col_LL; col_LR; col_RL];

ax = subplot(2,2,1); hold(ax,'on');
hb = boxplot(raw_plot, {'RR','LL','LR','RL'}, 'Colors', box_colors, 'Widths', boxWidth, 'Notch','off');
set(hb,'LineWidth',1);
xj = baseX + (rand(size(raw_plot,1),4)-0.5)*jitterWidth;

for c=1:4
    scatter(ax, xj(:,c), raw_plot(:,c), markerSize, 'MarkerEdgeColor','white', ...
        'MarkerFaceColor', box_colors(c,:), 'LineWidth', 0.1);
end
means_raw = nanmean(raw_plot,1);
for c=1:4
    plot(ax,[c-halfBW,c+halfBW],[means_raw(c) means_raw(c)],':','Color',box_colors(c,:),'LineWidth',1);
end
xlim(ax,[0.5,4.5]); ylim(ax,[1,10]); set(ax, 'YTick', 2:2:10);
set(ax,'FontSize',6);
ylabel(ax,'Peak Frequency (Hz)','FontSize',6);
title(ax,'Peak frequency by RT','FontSize',6);
box(ax,'off');

pdfFile = fullfile(outDir, 'Peak_frequency-RT_e1.pdf');
exportgraphics(gcf, pdfFile, 'ContentType','vector', 'BackgroundColor','none');

%% --- prepare for LMM ---
condNames = {'RR','LL','LR','RL'};
subjNames = string(subjects(validRowAny))';

T_z = array2table(z_plot, 'VariableNames', condNames);
T_z.subject = subjNames;
T_z = movevars(T_z, 'subject', 'Before', 1);

T = stack(T_z, condNames, 'NewDataVariableName','peak_z', 'IndexVariableName','condition');
T.sub_id    = cellfun(@(s) str2double(regexp(s,'\d+','match','once')), cellstr(T.subject));
T.subject   = string(T.subject);
T.condition = string(T.condition);

cs = string(T.condition);
T.hemi = cellstr(extractBetween(cs,1,1));
T.hand = cellstr(extractBetween(cs,2,2));
T      = T(:,{'subject','sub_id','condition','hemi','hand','peak_z'});

RT_mean = nan(numel(subjects), numel(conditions));
for i = 1:numel(subjects)
    Ti = allData{i,1}; if isempty(Ti); continue; end
    for ci = 1:numel(conditions)
        m = Ti.resptime(strcmp(Ti.condition, conditions{ci}));
        if ~isempty(m); RT_mean(i,ci) = mean(m,'omitnan'); end
    end
end

A = array2table(RT_mean(validRowAny,:), 'VariableNames', condNames);
A.subject = string(subjects(validRowAny))';
A = movevars(A, 'subject', 'Before', 1);
A = stack(A, condNames, 'NewDataVariableName','RT_mean', 'IndexVariableName','condition');
A.subject   = string(A.subject);
A.condition = string(A.condition);

T = innerjoin(T, A, 'Keys',{'subject','condition'});
T = T(:,{'subject','sub_id','condition','hemi','hand','peak_z','RT_mean'});
T = sortrows(T, 'sub_id');

outLongFile = fullfile(outDir, 'z_peak_fre_RT_long_e1.xlsx');
if exist(outLongFile,'file'); delete(outLongFile); end
writetable(T, outLongFile, 'Sheet','z_peak_long');

