import argparse
import math
import pandas as pd
import numpy
from scipy.stats import gmean
from numpy import mean

from bokeh.plotting import figure
from bokeh.io import curdoc, output_file, save
from bokeh.resources import CDN
from bokeh.embed import file_html, components
from bokeh.layouts import row, column
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, FactorRange
from bokeh.models.widgets import RadioButtonGroup
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from bokeh.events import ButtonClick

_FIELDS = ['Frequency of Reuse (Exact)',
           'Frequency of Reuse (0-0.1)',
           'Frequency of Reuse (0-0.25)',
           'No Comparison',
           'ANGER',
           'ANTICIPATION',
           'DISGUST',
           'FEAR',
           'JOY',
           'SADNESS',
           'SURPRISE',
           'TRUST',
           'NEGATIVE',
           'POSITIVE']

_AGG_FUNCS = [lambda x: gmean(x + 1) - 1] * 3
_AGG_FUNCS += [mean] * 11

# Possibly dead code now. TODO: Check and if so, remove.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--static', action='store_true',
                        default=False,
                        help="save a full html file")
    # parser.add_argument('-o', '--output', action='store',
    #                     default='reuse.html',
    #                     help="output filename")

    args = parser.parse_args()
    args.words_per_chunk = 140
    args.data_path = 'fandom-data.csv'
    title = 'Average Quantity of Text Reuse by {}-word Section'
    title = title.format(args.words_per_chunk)
    args.title = title
    args.out_filename = 'star-wars-reuse.html'
    return args

def word_formatter(names=None):
    if names is None:
        names = []

    punctuation = [',', '.', '!', '?', '\'', '"', ':', '-', '--']
    endpunctuation = ['.', '!', '?', '"', '...', '....', '--']
    contractions = ['\'ve', '\'m', '\'ll', '\'re', '\'s', '\'t', 'n\'t', 'na']
    capitals = ['i']

    def span(content, highlight=None):
        if highlight is None:
            return '<span>{}</span>'.format(content)
        else:
            style = 'background-color: rgba(16, 96, 255, {:04.3f})'.format(highlight)
            return '<span style="{}">{}</span>'.format(style, content)

    def format_word(word, prev_word, character, new_char, new_scene, highlight=None):
        parts = []
        if new_scene:
            parts.append(span('-- next scene--<br \>'))
        if new_char:
            parts.append('\n')
            parts.append(span(' ' + character.upper() + ': '))

        # Pandas annoyingly converts the string 'nan' into a floating
        # point nan value, even in an all-string column.
        if isinstance(word, float) and math.isnan(word):
            word = 'nan'

        if word in punctuation or word in contractions:
            # no space before punctuation
            parts.append(span(word, highlight))
        elif not prev_word or prev_word in endpunctuation:
            # capitalize first word of sentence
            parts.append(span(' ' + word.capitalize(), highlight))
        elif word in capitals:
            # format things like 'i'
            parts.append(span(' ' + word.upper(), highlight))
        elif word.capitalize() in names:
            # format names
            parts.append(span(' ' + word.capitalize(), highlight))
        else:
            # all other words
            parts.append(span(' ' + word, highlight))
        return ''.join(parts)
    return format_word

def chart_cols(fandom_data, words_per_chunk):
    words = fandom_data['LOWERCASE'].tolist()
    prevwords = [None] + words[:-1]
    chars = fandom_data['CHARACTER'].tolist()
    newchar = fandom_data['CHARACTER'][:-1].values != fandom_data['CHARACTER'][1:].values
    newchar = [True] + list(newchar)
    newscene = fandom_data['SCENE'].values
    newscene[numpy.isnan(newscene)] = 0
    newscene = fandom_data['SCENE'][:-1].values != fandom_data['SCENE'][1:].values
    newscene = [False] + list(newscene)


    highlights = fandom_data['Frequency of Reuse (Exact)'].tolist()
    chunks = (fandom_data.index // words_per_chunk).tolist()
    chunkmax = {}
    for h, c in zip(highlights, chunks):
        if c not in chunkmax or chunkmax[c] < h:
            chunkmax[c] = h
    highlights = [(h / chunkmax[c] if chunkmax[c] > 0 else 0)
                  for h, c in zip(highlights, chunks)]

    wform = word_formatter()
    spans = list(map(wform, words, prevwords, chars, newchar, newscene, highlights))

    fandom_data = fandom_data.assign(
        **{'No Comparison': fandom_data[_FIELDS[0]].values * 0}
    )
    chart_cols = fandom_data[_FIELDS]
    chart_cols = chart_cols.assign(chunk=chunks)
    chart_cols = chart_cols.assign(span=spans)

    return chart_cols

def join_wrap(seq):
    lines = []
    line = []
    last_br = 0
    for span in seq:
        if '\n' in span or last_br > 7 and '> ' in span:
            # Convert newlines to div breaks. Also insert breaks
            # whenever we've seen 7 words and there is some
            # leading whitespace in the current span.
            lines.append(''.join(line))
            line = []
            last_br = 0
        else:
            last_br += 1

        line.append(span)

    tail = ''.join(line)
    if tail.strip():
        lines.append(tail)

    return '\n'.join('<div>{}</div>'.format(l) for l in lines)

def chart_pivot(chart_cols):
    fields = _FIELDS + ['span']
    aggfuncs = _AGG_FUNCS + [join_wrap]
    return pd.pivot_table(
        chart_cols,
        values=fields,
        index=chart_cols.chunk,
        aggfunc=dict(zip(fields, aggfuncs))
    )

def build_bar_plot(data_path, words_per_chunk, title='Reuse'):
    #Read in from csv
    flat_data = pd.read_csv(data_path)
    flat_data = chart_cols(flat_data, words_per_chunk)
    flat_data = chart_pivot(flat_data)

    # Scale so that both maxima have the same height
    reuse_y = flat_data['Frequency of Reuse (Exact)']
    emo_y = flat_data['No Comparison']
    reuse_max = reuse_y.values.max()
    emo_max = emo_y.values.max()

    #Make ratio work
    ratio_denom = min(reuse_max, emo_max)
    ratio_num = max(reuse_max, emo_max)
    ratio = ratio_num / ratio_denom if ratio_denom > 0 else 1

    to_scale = reuse_y if reuse_max < emo_max else emo_y
    to_scale *= ratio

    # Create data columns
    grouped_x = [(str(x), key)
                 for x in flat_data.index
                 for key in ('Reuse', 'Emotion')]
    y = [re for re_pair in zip(reuse_y, emo_y) for re in re_pair]
    span = zip(flat_data.span, flat_data.span)
    span = [s for s_pair in span for s in s_pair]

    flat_data_source = ColumnDataSource(flat_data)
    source = ColumnDataSource(dict(x=grouped_x,
                                   y=y,
                                   span=span))

    plot = figure(x_range=FactorRange(*grouped_x),
                  plot_width=800, plot_height=600,
                  title=title, tools="hover")

    # Turn off ticks, major labels, and x grid lines, etc.
    # Axis settings:
    plot.xaxis.major_label_text_font_size = '0pt'
    plot.xaxis.major_tick_line_color = None
    plot.xaxis.minor_tick_line_color = None

    # CategoricalAxis settings:
    plot.xaxis.group_text_font_size = '0pt'
    plot.xaxis.separator_line_color = None

    # Grid settings:
    plot.xgrid.grid_line_color = None
    plot.ygrid.minor_grid_line_color = 'black'
    plot.ygrid.minor_grid_line_alpha = 0.03

    hover = plot.select(dict(type=HoverTool))
    hover.tooltips = "<div>@span{safe}</div>"
    plot.vbar(x='x',
              width=1.0,
              bottom=0,
              source=source,
              top='y',
              line_color='white',
              fill_color=factor_cmap('x', palette=Spectral6,
                                     factors=['Reuse', 'Emotion'],
                                     start=1, end=2))


    reuse_button_group = RadioButtonGroup(
        labels=_FIELDS[:3], button_type='primary'
        active=0
    )

    emotion_button_group = RadioButtonGroup(
        labels=_FIELDS[3:], button_type='success'
        active=0
    )

    callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group
        ),
        code="""
        var reuse = reuse_button_group.labels[reuse_button_group.active];
        var emo = emotion_button_group.labels[emotion_button_group.active];
        var reuse_data = flat_data_source.data[reuse].slice();  // Copy
        var emo_data = flat_data_source.data[emo].slice();      // Copy
        var reuse_max = Math.max.apply(Math, reuse_data);
        var emo_max = Math.max.apply(Math, emo_data);

        var ratio = 0;
        var to_scale = null;
        if (emo_max > reuse_max) {
            to_scale = reuse_data;
            ratio = emo_max / reuse_max;
        } else {
            to_scale = emo_data;
            if (emo_max > 0) {
                ratio = reuse_max / emo_max;
            } else {
                ratio = 1;
            }
        }
        for (var i = 0; i < to_scale.length; i++) {
            to_scale[i] *= ratio;
        }

        var x = source.data['x'];
        var y = source.data['y'];
        for (var i = 0; i < x.length; i++) {
            if (i % 2 === 0) {
                // This is a reuse bar
                y[i] = reuse_data[i / 2];
            } else {
                // This is an emotion bar
                y[i] = emo_data[(i - 1) / 2];
            }
        }
        source.change.emit();
        """
    )
    reuse_button_group.js_on_change('active', callback)
    emotion_button_group.js_on_change('active', callback)

    layout = column(reuse_button_group, emotion_button_group, plot)

    return layout

def build_line_plot(data_path, words_per_chunk, title='Reuse'):
    #Read in from csv
    flat_data = pd.read_csv(data_path)
    flat_data = chart_cols(flat_data, words_per_chunk)
    flat_data = chart_pivot(flat_data)

    # Scale so that both maxima have the same height
    reuse_y = flat_data['Frequency of Reuse (Exact)']
    emo_y = flat_data['No Comparison']
    reuse_max = reuse_y.values.max()
    emo_max = emo_y.values.max()
    third_y = emo_y

    #Make ratio work
    ratio_denom = min(reuse_max, emo_max)
    ratio_num = max(reuse_max, emo_max)
    ratio = ratio_num / ratio_denom if ratio_denom > 0 else 1

    to_scale = reuse_y if reuse_max < emo_max else emo_y
    to_scale *= ratio

    # Create data columns
    x = [str(i) for i in flat_data.index]
    reuse_zero = len(reuse_y) * [0]
    span = flat_data.span
    flat_data_source = ColumnDataSource(flat_data)
    source = ColumnDataSource(dict(x=x,
                                   emo_y=emo_y,
                                   reuse_zero=reuse_zero,
                                   reuse_y=reuse_y,
                                   third_y=third_y,
                                   span=span))

    plot = figure(x_range=FactorRange(*x),
                  plot_width=800, plot_height=600,
                  title=title, tools="hover")

    # Turn off ticks, major labels, and x grid lines, etc.
    # Axis settings:
    plot.xaxis.major_label_text_font_size = '0pt'
    plot.xaxis.major_tick_line_color = None
    plot.xaxis.minor_tick_line_color = None

    # CategoricalAxis settings:
    plot.xaxis.group_text_font_size = '0pt'
    plot.xaxis.separator_line_color = None

    # Grid settings:
    plot.xgrid.grid_line_color = None
    plot.ygrid.minor_grid_line_color = 'black'
    plot.ygrid.minor_grid_line_alpha = 0.03

    hover = plot.select(dict(type=HoverTool))
    hover.tooltips = "<div>@span{safe}</div>"

    plot.varea(x='x', source = source, y1 = 'reuse_y', y2 = 'reuse_zero', fill_color = Spectral6[0], fill_alpha = 0.6)
    plot.line(x='x', line_width=2.0, source=source, y='emo_y', line_color = Spectral6[1])
    plot.line(x='x', line_width=2.0, source=source, y='third_y', line_color = 'red')

    reuse_button_group = RadioButtonGroup(
        labels=_FIELDS[:3],
        active=0,
        button_type='primary'
    )

    emotion_button_group = RadioButtonGroup(
        labels=_FIELDS[3:],
        active=0,
        button_type='success'
    )

    third_button_group = RadioButtonGroup(
        labels=_FIELDS[3:],
        active=0,
        button_type='danger'
    )

    callback_code="""
        var reuse = reuse_button_group.labels[reuse_button_group.active];
        var emo = emotion_button_group.labels[emotion_button_group.active];
        var third = third_button_group.labels[third_button_group.active];
        var reuse_data = flat_data_source.data[reuse].slice();  // Copy
        var emo_data = flat_data_source.data[emo].slice();      // Copy
        var third_data = flat_data_source.data[third].slice();
        var reuse_max = Math.max.apply(Math, reuse_data);
        var emo_max = Math.max.apply(Math, emo_data);

        var ratio = 0;
        var to_scale = null;
        if (emo_max > reuse_max) {
            to_scale = reuse_data;
            ratio = emo_max / reuse_max;
        } else {
            to_scale = emo_data;
            if (emo_max > 0) {
                ratio = reuse_max / emo_max;
            } else {
                ratio = 1;
            }
        }
        // for (var i = 0; i < to_scale.length; i++) {
        //    to_scale[i] *= ratio;
        // }

        var x = source.data['x'];
        var reuse_y = source.data['reuse_y'];
        var emo_y = source.data['emo_y']
        var third_y = source.data['third_y']
        for (var i = 0; i < x.length; i++) {
            reuse_y[i] = reuse_data[i];
            emo_y[i] = emo_data[i];
            third_y[i] = third_data[i];
        }

        source.change.emit();
        if (third_button_group.active == 0 || emotion_button_group.active == 0) {
            return;
        }

        if (other_button_group) {
            other_button_group.active = 0;
        }
    """

    reuse_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group,
            third_button_group=third_button_group,
            other_button_group=None
        ),
        code=callback_code
    )

    emo_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group,
            third_button_group=third_button_group,
            other_button_group=third_button_group
        ),
        code=callback_code
    )

    third_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group,
            third_button_group=third_button_group,
            other_button_group=emotion_button_group
        ),
        code=callback_code
    )

    reuse_button_group.js_on_click(reuse_callback)
    emotion_button_group.js_on_click(emo_callback)
    third_button_group.js_on_click(third_callback)

    layout = column(reuse_button_group, emotion_button_group, third_button_group, plot)

    return layout

def build_plot(args):
    if args.lineplot:
        return build_line_plot(args.input, args.words_per_chunk)
    else:
        return build_bar_plot(args.input, args.words_per_chunk)

def save_static(args):
    plot = build_plot(args)
    file_html(plot, CDN, args.title)
    output_file(args.output,
                title=args.title, mode="cdn")
    save(plot)

def save_embed(args):
    plot = build_plot(args)
    with open(args.output, 'w', encoding='utf-8') as op:
        for c in components(plot):
            op.write(c)
            op.write('\n')

def save_plot(args):
    title = 'Average Quantity of Text Reuse by {}-word Section'
    title = title.format(args.words_per_chunk)
    args.title = title

    if args.static:
        save_static(args)
    else:
        save_embed(args)
