# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals


from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
import six
# End: Python 2/3 compatability header small


###############################################################################
###############################################################################
###############################################################################


import imp
import matplotlib.pyplot as plt
import numpy as np
import os
import keras
import innvestigate
from innvestigate.applications import imagenet

base_dir = os.path.dirname(__file__)
utils = imp.load_source("utils", os.path.join(base_dir, "utils.py"))
imgnetutils = imp.load_source("utils_imagenet", os.path.join(base_dir, "utils_imagenet.py"))


###############################################################################
###############################################################################
###############################################################################

METHODS = {
    'input': {},
    'pattern.net': {'net_args': ['patterns'],
                    'f_visualize': imgnetutils.bk_proj
                    },
    'pattern.attribution': {'net_args': ['patterns'],
                            'f_visualize': imgnetutils.heatmap
                            },
    'gradient': {#'kwargs': {'postprocess': 'abs'},
                 'f_visualize': imgnetutils.heatmap # imgnetutils.graymap #imgnetutils.bk_proj
                 },
    'input_t_gradient': {#'kwargs': {'postprocess': 'abs'},
                         'f_visualize': imgnetutils.heatmap
                         },
    'pattern.net*gradient': {'f_visualize': imgnetutils.heatmap
                             },
    'input*gradient': {'f_visualize': imgnetutils.heatmap
                      },
    'deep_taylor.bounded': {'net_getter_kwargs': {'low': lambda net: net["input_range"][0],
                                                  'high': lambda net: net["input_range"][1]},
                            'f_visualize': imgnetutils.heatmap}
}

def create_vgg16_model_and_analyzer(methods_metadata, analyzer_names=('input', 'pattern.net',)):
    # Get model and various metadata
    net = imagenet.vgg16(load_weights=True, load_patterns="relu")

    # Build the model.
    model = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    # Strip softmax layer
    model_wo_softmax = innvestigate.utils.model_wo_softmax(model)

    def _create_analyzer(analyzer_name):
        meta = methods_metadata[analyzer_name]
        if 'net_args' in meta:
            a_kwargs = {net_arg_name: net[net_arg_name] for net_arg_name in meta['net_args']}
        else:
            a_kwargs = {}
        if 'kwargs' in meta:
            a_kwargs.update(meta['kwargs'])
        if 'net_getter_kwargs' in meta:
            for arg_name in meta['net_getter_kwargs']:
                arg_func = meta['net_getter_kwargs'][arg_name]
                a_kwargs[arg_name] = arg_func(net)
        return innvestigate.create_analyzer(analyzer_name, model_wo_softmax, **a_kwargs)


    # Create analyzers
    analyzers = []
    for an in analyzer_names:
        if '*' in an:
            sub_analyzer_names = an.split('*')
            analyzer = [_create_analyzer(sub_an) for sub_an in sub_analyzer_names]
            an = sub_analyzer_names
        else:
            analyzer = _create_analyzer(an)
        analyzers.append((an, analyzer))


    return model, net, analyzers


# adapted from examples/notebook/imagenet_compare_methods.ipnb
def analyze(data, analyzers, net, model, label_to_class_name):

    # Strip softmax layer
    model_wo_softmax = innvestigate.utils.model_wo_softmax(model)

    # Handle input depending on model and backend.
    channels_first = keras.backend.image_data_format() == "channels_first"
    color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None

    analysis = np.zeros([len(data), len(analyzers)] + net["image_shape"] + [3])
    text = []

    for i, (x, y) in enumerate(data):
        # Add batch axis.
        x = x[None, :, :, :]
        x_pp = imgnetutils.preprocess(x, net)

        # Predict final activations, probabilities, and label.
        presm = model_wo_softmax.predict_on_batch(x_pp)[0]
        prob = model.predict_on_batch(x_pp)[0]
        y_hat = prob.argmax()

        # Save prediction info:
        text.append(("%s" % label_to_class_name[y],  # ground truth label
                     "%.2f" % presm.max(),  # pre-softmax logits
                     "%.2f" % prob.max(),  # probabilistic softmax output
                     "%s" % label_to_class_name[y_hat]  # predicted label
                     ))

        for aidx, (method_name, analyzer) in enumerate(analyzers):
            if method_name == 'input':
                # Do not analyze, but keep not preprocessed input.
                a = x / 255
            elif type(analyzer) is list or type(analyzer) is tuple:
                a_subs = []
                for sub_method_name, sub_analyzer in zip(method_name, analyzer):
                    # Analyze.
                    a = sub_analyzer.analyze(x_pp)

                    # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                    a = imgnetutils.postprocess(a, color_conversion, channels_first)
                    a_subs.append(a)
                assert len(a_subs) > 0, 'no sub-analyzer applied'
                a = a_subs[0]
                for a_next in a_subs[1:]:
                    a = a * a_next

            elif analyzer:
                # Analyze.
                a = analyzer.analyze(x_pp)

                # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                a = imgnetutils.postprocess(a, color_conversion, channels_first)

            else:
                a = np.zeros_like(x)

            # Store the analysis.
            analysis[i, aidx] = a[0]

    return analysis, text

# adapted from examples/notebook/imagenet_compare_methods.ipnb
def visualize(analysis, text, method_names, methods_metadata):
    # Apply analysis postprocessing, e.g., creating a heatmap.
    #postprocess = {i: METHODS[method_name].get('visualize', lambda x: x) for i, method_name in enumerate(method_names)}
    # Prepare the grid as rectangular list
    #grid = [[analysis[i, j] for j in range(analysis.shape[1])]
    #        for i in range(analysis.shape[0])]
    grid = []
    for i in range(analysis.shape[0]):
        grid.append([])
        for j in range(analysis.shape[1]):
            _method_name = method_names[j]
            #_p_func = METHODS[_method_name].get('f_visualize', lambda x: x)
            grid[i].append(analysis[i, j])
            if 'f_visualize' in methods_metadata[_method_name]:
                _p_func = methods_metadata[_method_name]['f_visualize']
                # batch dim has to be added before f_visualize application (otherwise bk_proj fails)
                grid[i][j] = _p_func(np.expand_dims(grid[i][j], axis=0))[0]



    # Prepare the labels
    label, presm, prob, pred = zip(*text)
    row_labels_left = [('label: {}'.format(label[i]), 'pred: {}'.format(pred[i])) for i in range(len(label))]
    row_labels_right = [('logit: {}'.format(presm[i]), 'prob: {}'.format(prob[i])) for i in range(len(label))]
    col_labels = [''.join(method_name) for method_name in method_names]

    # Plot the analysis.
    utils.plot_image_grid(grid, row_labels_left, row_labels_right, col_labels,
                        #file_name=os.environ.get("plot_file_name", None),
                          file_name='test_analysis'
                          )

def main():

    # To use this script please download the example images using the following script:
    # innvestigate/examples/images/wget_imagenet_2011_samples.sh
    #method_names=['input', 'pattern.net',]
    #method_names = ['input', 'gradient', 'pattern.net*gradient']
    method_names = ['input', 'gradient', 'deep_taylor.bounded', 'pattern.attribution', 'pattern.net', 'input_t_gradient']# 'input*gradient', 'pattern.net*gradient', ]
    model, net, analyzers = create_vgg16_model_and_analyzer(methods_metadata=METHODS,
        analyzer_names=method_names

    )

    images, label_to_class_name = utils.get_imagenet_data(net["image_shape"][0])

    if not len(images):
        raise Exception("Please download the example images using: "
                        "'innvestigate/examples/images/wget_imagenet_2011_samples.sh'")

    analysis, text = analyze(data=images, analyzers=analyzers, net=net, model=model, label_to_class_name=label_to_class_name)
    #visualize(analysis, text, method_names)
    _get_analysis = {method_name: np.expand_dims(analysis[:,idx], axis=1) for idx, method_name in enumerate(method_names)}
    gradient_input = _get_analysis['gradient'] * _get_analysis['input']
    gradient_patternnet = _get_analysis['gradient'] * _get_analysis['pattern.net']
    analysis = np.concatenate((analysis, gradient_input, gradient_patternnet), axis=1)
    visualize(analysis, text, method_names + ['input*gradient', 'pattern.net*gradient'], methods_metadata=METHODS)

if __name__ == "__main__":
    main()
