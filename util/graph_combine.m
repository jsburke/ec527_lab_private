function graph_combine(file)

data = csvread(file,1,0);

%processing to read column headers
fid = fopen(file);
legend_names = strsplit(fgetl(fid),',');
legend_names = legend_names(2:end);
fclose(fid);

figure; hold on;
plots = [];
legs  = [];

colors  = ['b','r','g','m','k'];   % colors for plots
markers = [' ','o','+','s','*','x'];   % line markers for plots
color_index  = 1;
color_len = length(colors);
markers_index = 1;
x_axis = data(:,1);               %get x values to plot against
num_cols = size(data,2);          %get number of y value columns
for i = 2:num_cols
    if markers_index == 1
       plot_feature = strcat(colors(color_index), '-');
    else
       plot_feature = strcat(colors(color_index), '-', markers(markers_index));
    end
    
    %  Where the plotting actually happens
    cur_plot = plot(x_axis, data(:,i), plot_feature);
    cur_leg  = legend_names(i-1);
    
    if color_index == color_len
       color_index = 1; 
       markers_index = markers_index + 1;  % hopefully we'll never need more 
    else
       color_index = color_index + 1;
    end
    plots  = [plots ; cur_plot];
    legs = [legs ; cur_leg];
end
legend(plots, legs, 'LOCATION', [0.15, 0.65, .25, .25]);
