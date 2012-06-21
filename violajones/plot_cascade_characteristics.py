
from violajones.parse_haar import parse_haar_xml

def calc_norm_feature_size(feature, cascade_size):
        minx = float('inf')
        miny = float('inf')
        maxx = -float('inf')
        maxy = -float('inf')
        for (x, y, width, height), _ in feature.shapes:
            minx = min(x, minx)
            miny = min(y, miny)
            maxx = max(x+width, maxx)
            maxy = max(y+height, maxy)
        feature_width = float(maxx - minx)
        feature_height = float(maxy - miny)
        return feature_width/cascade_size[0] * feature_height/cascade_size[1]


def average_norm_feature_size(stage, cascade_size):
        return sum(calc_norm_feature_size(f, cascade_size) for f in stage.features)/len(stage.features)

if __name__ == '__main__':
        import sys
        if len(sys.argv) < 2:
                print 'usage: %s cascadefile'%sys.argv[0]
                exit(1)
        opt_args = sys.argv[2:]
        save_plots = '--save-plots' in opt_args
        disable_show_plots = '--disable-show-plots' in opt_args
        cascade_name = sys.argv[1]
        cascade = parse_haar_xml(cascade_name)
        print cascade

        stages = range(len(cascade.stages))
        nr_features = [len(stage.features) for stage in cascade.stages]
        avg_feature_size = [average_norm_feature_size(stage, cascade.size) for stage in cascade.stages]

        print stages
        print nr_features
        print avg_feature_size


        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from matplotlib.backends.backend_pdf import PdfPages
        import numpy as np

        plot_values = [\
            ('Number of features', nr_features),
            ('Average feature area', avg_feature_size),
        ]

        name = cascade_name.rsplit('/')[-1]
        #linestyle = ['k--', 'k:', 'k', 'k-']
        linestyle = ['*-', 'k:', 'k', 'k-']
        for i, (name, values) in enumerate(plot_values):
                fig = plt.figure()
                fig.set_label(name)
                fig.canvas.manager.set_window_title(name)
                ax = fig.add_subplot(111)
                #ax.set_color_cycle(['r', 'g', 'b', 'c'])
                lines = []
                lines.append(stages)
                lines.append(values)
                lines.append(linestyle[0])
                ax.plot(*lines)
                ax.set_xlabel('cascade stage')
                ax.set_ylabel(name)
                ax.set_xticklabels([])
                ax.set_yticklabels([])

                if save_plots:
                    try:
                        p = PdfPages(name + '.pdf')
                        p.savefig()
                        p.close()
                    except Exception, e:
                        print 'could not save graph to pdf file'
                        print 'error:', str(e)

        # finally show all plots
        if not disable_show_plots:
            plt.show()
            
