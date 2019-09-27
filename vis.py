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
from bokeh.models import HoverTool, CustomJS, ColumnDataSource, FactorRange, Panel, Tabs
from bokeh.models.widgets import RadioButtonGroup, Select
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
from bokeh.events import ButtonClick

_FIELDS = ['Frequency of Reuse (Exact Matches)',
           'Frequency of Reuse (0-0.1)',
           'Frequency of Reuse (0-0.25)',
           'None',
           'ANGER',
           'ANTICIPATION',
           'DISGUST',
           'FEAR',
           'JOY',
           'SADNESS',
           'SURPRISE',
           'TRUST',
           'NEGATIVE',
           'POSITIVE',]
           # 'None']

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


    highlights = fandom_data['Frequency of Reuse (Exact Matches)'].tolist()
    chunks = (fandom_data.index // words_per_chunk).tolist()
    chunkmax = {}
    for h, c in zip(highlights, chunks):
        if c not in chunkmax or chunkmax[c] < h:
            chunkmax[c] = h
    highlights = [( h / (1.6 * chunkmax[c]) if chunkmax[c] > 0 else 0)
                  for h, c in zip(highlights, chunks)]

    wform = word_formatter()
    spans = list(map(wform, words, prevwords, chars, newchar, newscene, highlights))

    fandom_data = fandom_data.assign(
        **{'None': fandom_data[_FIELDS[0]].values * 0}
    )

    # fandom_data = fandom_data.assign(
    #     **{'None': fandom_data[_FIELDS[0]].values * 0}
    # )

    character_cols = [x for x in fandom_data.columns if x.startswith("CHARACTER_")]
    chart_cols = fandom_data[_FIELDS + character_cols]
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
    character_cols = [x for x in chart_cols.columns if x.startswith("CHARACTER_")]
    print(character_cols)
    fields = _FIELDS + character_cols + ['span']
    aggfuncs = _AGG_FUNCS + [mean] * len(character_cols) + [join_wrap]
    table =  pd.pivot_table(
        chart_cols,
        values=fields,
        index=chart_cols.chunk,
        aggfunc=dict(zip(fields, aggfuncs))
    )
    # apparently when you create a pandas pivot table, it will automatically
    # sort your columns alphabetically (which is dumb). This is their work
    # around, where you literally give the table the fields you already gave
    # them, so that they "reindex" it.
    return table.reindex(fields, axis=1)


def build_bar_plot(data_path, words_per_chunk, title='Reuse'):
    #Read in from csv
    flat_data = pd.read_csv(data_path)
    flat_data = chart_cols(flat_data, words_per_chunk)
    flat_data = chart_pivot(flat_data)

    # Scale so that both maxima have the same height
    reuse_y = flat_data['Frequency of Reuse (Exact Matches)']
    emo_y = flat_data['None']
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
        labels= [_FIELDS[0]] + ["Frequency of Reuse (Fuzzy Matches)"], button_type='primary',
        active=0
    )

    emotion_button_group = RadioButtonGroup(
        labels=_FIELDS[3:], button_type = "success",
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
        if (reuse == "Frequency of Reuse (Fuzzy Matches)") {
            reuse = "Frequency of Reuse (0-0.25)";
        }
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
    tab1 = Panel(child=layout, title='Bar')
    return tab1

def build_line_plot(data_path, words_per_chunk, title='Reuse'):
    #Read in from csv
    flat_data = pd.read_csv(data_path)
    flat_data = chart_cols(flat_data, words_per_chunk)
    flat_data = chart_pivot(flat_data)

    # Scale so that both maxima have the same height
    reuse_y = flat_data['Frequency of Reuse (Exact Matches)']
    emo_y = flat_data['None']
    char_y = flat_data['None']
    reuse_max = reuse_y.values.max()
    emo_max = emo_y.values.max()
    char_max = char_y.values.max()

    #Make ratio work
    ratio_denom = min(char_max, min(reuse_max, emo_max))
    ratio_num = max(char_max, max(reuse_max, emo_max))
    ratio = ratio_num / ratio_denom if ratio_denom > 0 else 1
    if reuse_max < emo_max and reuse_max < char_max:
        to_scale = reuse_y
    elif emo_max < char_max and emo_max < reuse_max:
        to_scale = emo_y
    else:
        to_scale = char_y
    to_scale *= ratio

    # Create data columns
    x = [str(i) for i in flat_data.index]
    reuse_y=reuse_y
    reuse_zero = len(reuse_y) * [0]
    span = flat_data.span
    flat_data_source = ColumnDataSource(flat_data)
    source = ColumnDataSource(dict(x=x,
                                   reuse_y=reuse_y,
                                   emo_y=emo_y,
                                   char_y=char_y,
                                   reuse_zero=reuse_zero,
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
    plot.line(x='x', source = source, y = 'reuse_y', line_color = Spectral6[0], line_alpha = 0.0)
    plot.line(x='x', line_width=2.0, source=source, y='emo_y', line_color = Spectral6[1])
    plot.line(x='x', line_width=2.0, source=source, y='char_y', line_color = 'red')


    reuse_button_group = RadioButtonGroup(
        labels=[_FIELDS[0]] + ["Frequency of Reuse (Fuzzy Matches)"],
        button_type='primary',
        active=0
    )

    emotion_button_group = RadioButtonGroup(
        labels=_FIELDS[3:],
        button_type='success',
        active=0
    )

    char_button_group = RadioButtonGroup(
        labels= ['None'] + [x.replace("CHARACTER_", "") for x in flat_data.columns if x.startswith("CHARACTER_")],
        button_type='danger',
        active=0
    )

    callback_code="""
        var reuse = reuse_button_group.labels[reuse_button_group.active];
        if (reuse == "Frequency of Reuse (Fuzzy Matches)") {
            reuse = "Frequency of Reuse (0-0.25)";
        }
        var emo = emotion_button_group.labels[emotion_button_group.active];
        var char = char_button_group.labels[char_button_group.active];
        var reuse_data = flat_data_source.data[reuse].slice();  // Copy
        var emo_data = flat_data_source.data[emo].slice();      // Copy
        if (char == "None") {
            var char_data = flat_data_source.data["None"].slice();
        } else {
            var char_data = flat_data_source.data["CHARACTER_" + char].slice();
            }  // Copy
        var reuse_max = Math.max.apply(Math, reuse_data);
        var emo_max = Math.max.apply(Math, emo_data);
        var char_max = Math.max.apply(Math, char_data);

        var ratio = 0;
        var to_scale = null;
        var to_scale_other = null;

        if (emo_max > reuse_max && emo_max > char_max) {
            to_scale = reuse_data;
            to_scale_also = char_data;
            ratio_one = emo_max / reuse_max;
            ratio_two = emo_max / char_max;
        } else if (char_max > emo_max && char_max > reuse_max) {
            to_scale = reuse_data;
            to_scale_also = emo_data;
            ratio_one = char_max / reuse_max;
            ratio_two = char_max / emo_max;
        } else {
            to_scale = emo_data;
            to_scale_also = char_data;
            ratio_one = reuse_max / emo_max;
            ratio_two = reuse_max / char_max;
        }

        for (var i = 0; i < to_scale.length; i++) {
            to_scale[i] *= ratio_one;
            to_scale_also[i] *= ratio_two;
        }

        var x = source.data['x'];
        var reuse_y = source.data['reuse_y'];
        var emo_y = source.data['emo_y'];
        var char_y = source.data['char_y']
        for (var i = 0; i < x.length; i++) {
            reuse_y[i] = reuse_data[i];
            emo_y[i] = emo_data[i];
            char_y[i] = char_data[i];
        }

        source.change.emit();
        if (char_button_group.active == 0 || emotion_button_group.active == 0) {
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
            char_button_group=char_button_group,
            other_button_group=None
        ), code = callback_code)


    emo_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group,
            char_button_group=char_button_group,
            other_button_group=char_button_group
        ), code = callback_code)

    char_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group,
            char_button_group=char_button_group,
            other_button_group=emotion_button_group
        ), code = callback_code)


    reuse_button_group.js_on_change('active', reuse_callback)
    emotion_button_group.js_on_change('active', emo_callback)
    char_button_group.js_on_change('active', char_callback)


    layout = column(reuse_button_group, emotion_button_group, char_button_group, plot)
    tab1 = Panel(child=layout, title='Line')
    return tab1


def build_line_plot_compare(data_path, words_per_chunk, title='Degree of Reuse'):
    #Read in from csv
    flat_data = pd.read_csv(data_path)

    flat_data = chart_cols(flat_data, words_per_chunk)

    flat_data = chart_pivot(flat_data)


    # Scale so that both maxima have the same height
    reuse_y = flat_data['Frequency of Reuse (Exact Matches)']
    emo_y = flat_data['None']
    char_y = flat_data['None']
    reuse_max = reuse_y.values.max()
    emo_max = emo_y.values.max()
    char_max = char_y.values.max()

    #Make ratio work
    ratio_denom = min(char_max, min(reuse_max, emo_max))
    ratio_num = max(char_max, max(reuse_max, emo_max))
    ratio = ratio_num / ratio_denom if ratio_denom > 0 else 1
    if reuse_max < emo_max and reuse_max < char_max:
        to_scale = reuse_y
    elif emo_max < char_max and emo_max < reuse_max:
        to_scale = emo_y
    else:
        to_scale = char_y
    to_scale *= ratio

    # Create data columns
    x = [str(i) for i in flat_data.index]
    reuse_y=reuse_y
    reuse_zero = len(reuse_y) * [0]
    span = flat_data.span
    flat_data_source = ColumnDataSource(flat_data)
    source = ColumnDataSource(dict(x=x,
                                   reuse_y=reuse_y,
                                   emo_y=emo_y,
                                   char_y=char_y,
                                   reuse_zero=reuse_zero,
                                   span=span))

    plot = figure(x_range=FactorRange(*x),
                  plot_width=800, plot_height=600,
                  title=title, tools="hover")

    # Turn off ticks, major labels, and x grid lines, etc.
    # Axis settings:
    plot.xaxis.major_label_text_font_size = '0pt'
    plot.xaxis.major_tick_line_color = None
    plot.xaxis.minor_tick_line_color = None

    plot.yaxis.major_label_text_font_size = '0pt'
    plot.yaxis.major_tick_line_color = None
    plot.yaxis.minor_tick_line_color = None

    # CategoricalAxis settings:
    plot.xaxis.group_text_font_size = '0pt'
    plot.xaxis.separator_line_color = None

    # Grid settings:
    plot.xgrid.grid_line_color = None
    # plot.ygrid.minor_grid_line_color = 'black'
    # plot.ygrid.minor_grid_line_alpha = 0.03
    plot.xaxis.axis_label = 'Beginning of Script    ←                                                                                                                                      →   End of Script'
    plot.yaxis.axis_label = 'Low Reuse                           Medium Reuse                                  High Reuse'

    hover = plot.select(dict(type=HoverTool))
    hover.tooltips = "<div>@span{safe}</div>"

    plot.varea(x='x', source = source, y1 = 'reuse_y', y2 = 'reuse_zero', fill_color = Spectral6[0], fill_alpha = 0.6)
    plot.line(x='x', source = source, y = 'reuse_y', line_color = Spectral6[0], line_alpha = 0.0)
    plot.line(x='x', line_width=2.0, source=source, y='emo_y', line_color = Spectral6[1])
    plot.line(x='x', line_width=2.0, source=source, y='char_y', line_color = 'red')


    reuse_button_group = RadioButtonGroup(
        labels=[_FIELDS[0]] + ["Frequency of Reuse (Fuzzy Matches)"],
        button_type='primary',
        active=0
    )

    emotion_button_group = RadioButtonGroup(
        labels=_FIELDS[3:],
        button_type='success',
        active=0
    )

    char_button_group = RadioButtonGroup(
        labels= ['None'] + [x.replace("CHARACTER_", "") for x in flat_data.columns if x.startswith("CHARACTER_")],
        button_type='danger',
        active=0
    )

    print(char_button_group.labels)


    callback_code="""
        var reuse = reuse_button_group.labels[reuse_button_group.active];
        if (reuse == "Frequency of Reuse (Fuzzy Matches)") {
            reuse = "Frequency of Reuse (0-0.25)";
        }
        var emo = emotion_button_group.labels[emotion_button_group.active];
        var char = char_button_group.labels[char_button_group.active];
        var reuse_data = flat_data_source.data[reuse].slice();  // Copy
        var emo_data = flat_data_source.data[emo].slice();      // Copy
        if (char == "None") {
            var char_data = flat_data_source.data["None"].slice();
        } else {
            var char_data = flat_data_source.data["CHARACTER_" + char].slice();
            }  // Copy
        var reuse_max = Math.max.apply(Math, reuse_data);
        var emo_max = Math.max.apply(Math, emo_data);
        var char_max = Math.max.apply(Math, char_data);

        var ratio = 0;
        var to_scale = null;
        var to_scale_other = null;

        if (emo_max > reuse_max && emo_max > char_max) {
            to_scale = reuse_data;
            to_scale_also = char_data;
            ratio_one = emo_max / reuse_max;
            ratio_two = emo_max / char_max;
        } else if (char_max > emo_max && char_max > reuse_max) {
            to_scale = reuse_data;
            to_scale_also = emo_data;
            ratio_one = char_max / reuse_max;
            ratio_two = char_max / emo_max;
        } else {
            to_scale = emo_data;
            to_scale_also = char_data;
            ratio_one = reuse_max / emo_max;
            ratio_two = reuse_max / char_max;
        }

        for (var i = 0; i < to_scale.length; i++) {
            to_scale[i] *= ratio_one;
            to_scale_also[i] *= ratio_two;
        }

        var x = source.data['x'];
        var reuse_y = source.data['reuse_y'];
        var emo_y = source.data['emo_y'];
        var char_y = source.data['char_y']
        for (var i = 0; i < x.length; i++) {
            reuse_y[i] = reuse_data[i];
            emo_y[i] = emo_data[i];
            char_y[i] = char_data[i];
        }

        source.change.emit();

        """

    reuse_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group,
            char_button_group=char_button_group,
            other_button_group=None
        ), code = callback_code)


    emo_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group,
            char_button_group=char_button_group,
            other_button_group=char_button_group
        ), code = callback_code)

    char_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_button_group=emotion_button_group,
            char_button_group=char_button_group,
            other_button_group=emotion_button_group
        ), code = callback_code)


    reuse_button_group.js_on_change('active', reuse_callback)
    emotion_button_group.js_on_change('active', emo_callback)
    char_button_group.js_on_change('active', char_callback)


    layout = column(reuse_button_group, emotion_button_group, char_button_group, plot)
    # tab1 = Panel(child=layout, title='Compare')
    # return tab1
    return layout

def build_line_plot_dropdown(data_path, words_per_chunk, title='Reuse'):
    #Read in from csv
    flat_data = pd.read_csv(data_path)
    flat_data = chart_cols(flat_data, words_per_chunk)
    flat_data = chart_pivot(flat_data)

    # Scale so that both maxima have the same height
    reuse_y = flat_data['Frequency of Reuse (Exact Matches)']
    emo_y = flat_data['None']
    char_y = flat_data['None']
    reuse_max = reuse_y.values.max()
    emo_max = emo_y.values.max()
    char_max = char_y.values.max()

    #Make ratio work
    ratio_denom = min(char_max, min(reuse_max, emo_max))
    ratio_num = max(char_max, max(reuse_max, emo_max))
    ratio = ratio_num / ratio_denom if ratio_denom > 0 else 1
    if reuse_max < emo_max and reuse_max < char_max:
        to_scale = reuse_y
    elif emo_max < char_max and emo_max < reuse_max:
        to_scale = emo_y
    else:
        to_scale = char_y
    to_scale *= ratio

    # Create data columns
    x = [str(i) for i in flat_data.index]
    reuse_y=reuse_y
    reuse_zero = len(reuse_y) * [0]
    span = flat_data.span
    flat_data_source = ColumnDataSource(flat_data)
    source = ColumnDataSource(dict(x=x,
                                   reuse_y=reuse_y,
                                   emo_y=emo_y,
                                   char_y=char_y,
                                   reuse_zero=reuse_zero,
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
    plot.line(x='x', source = source, y = 'reuse_y', line_color = Spectral6[0], line_alpha = 0.0)
    plot.line(x='x', line_width=2.0, source=source, y='emo_y', line_color = Spectral6[1])
    plot.line(x='x', line_width=2.0, source=source, y='char_y', line_color = 'red')


    reuse_button_group = RadioButtonGroup(
        labels= [_FIELDS[0]] + ["Frequency of Reuse (Fuzzy Matches)"], button_type='primary',
        active=0
    )

    emotion_dropdown_button_group = Select(
           title="Emotion", value="None", options=_FIELDS[3:])

    char_dropdown_button_group = Select(
           title="Emotion2", value="None", options=_FIELDS[3:])

    callback_code="""
        var reuse = reuse_button_group.labels[reuse_button_group.active];
        if (reuse == "Frequency of Reuse (Fuzzy Matches)") {
            reuse = "Frequency of Reuse (0-0.25)";
        }
        var emo = emotion_dropdown_button_group.value;
        var char = char_dropdown_button_group.value;
        var reuse_data = flat_data_source.data[reuse].slice();  // Copy
        var emo_data = flat_data_source.data[emo].slice();      // Copy
        var char_data = flat_data_source.data[char].slice();      // Copy
        var reuse_max = Math.max.apply(Math, reuse_data);
        var emo_max = Math.max.apply(Math, emo_data);
        var char_max = Math.max.apply(Math, char_data);

        var ratio = 0;
        var to_scale = null;
        var to_scale_other = null;

        if (emo_max > reuse_max && emo_max > char_max) {
            to_scale = reuse_data;
            to_scale_also = char_data;
            ratio_one = emo_max / reuse_max;
            ratio_two = emo_max / char_max;
        } else if (char_max > emo_max && char_max > reuse_max) {
            to_scale = reuse_data;
            to_scale_also = emo_data;
            ratio_one = char_max / reuse_max;
            ratio_two = char_max / emo_max;
        } else {
            to_scale = emo_data;
            to_scale_also = char_data;
            ratio_one = reuse_max / emo_max;
            ratio_two = reuse_max / char_max;
        }

        for (var i = 0; i < to_scale.length; i++) {
            to_scale[i] *= ratio_one;
            to_scale_also[i] *= ratio_two;
        }

        var x = source.data['x'];
        var reuse_y = source.data['reuse_y'];
        var emo_y = source.data['emo_y'];
        var char_y = source.data['char_y']
        for (var i = 0; i < x.length; i++) {
            reuse_y[i] = reuse_data[i];
            emo_y[i] = emo_data[i];
            char_y[i] = char_data[i];
        }

        source.change.emit();
        if (char_dropdown_button_group.value == "None" || emotion_dropdown_button_group.value == "None") {
            return;
        }

        if (other_button_group) {
            other_button_group.value = "None";
        }

        """

    reuse_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_dropdown_button_group=emotion_dropdown_button_group,
            char_dropdown_button_group=char_dropdown_button_group,
            other_button_group=None
        ), code = callback_code)


    emo_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_dropdown_button_group=emotion_dropdown_button_group,
            char_dropdown_button_group=char_dropdown_button_group,
            other_button_group=char_dropdown_button_group
        ), code = callback_code)

    char_callback = CustomJS(
        args=dict(
            source=source,
            flat_data_source=flat_data_source,
            reuse_button_group=reuse_button_group,
            emotion_dropdown_button_group=emotion_dropdown_button_group,
            char_dropdown_button_group=char_dropdown_button_group,
            other_button_group=emotion_dropdown_button_group
        ), code = callback_code)


    reuse_button_group.js_on_change('active', reuse_callback)
    emotion_dropdown_button_group.js_on_change('value', emo_callback)
    char_dropdown_button_group.js_on_change('value', char_callback)


    layout = column(reuse_button_group, emotion_dropdown_button_group, char_dropdown_button_group, plot)
    tab1 = Panel(child=layout, title='Line Dropdown')
    return tab1

def build_plot(args):
    # return Tabs(tabs=[build_line_plot(args.input, args.words_per_chunk),
    #                   # build_line_plot_dropdown(args.input, args.words_per_chunk),
    #                   build_line_plot_compare(args.input, args.words_per_chunk)])
    return build_line_plot_compare(args.input, args.words_per_chunk)



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
