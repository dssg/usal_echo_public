def get_views_to_indices(model):
    """ Get dictionary of view name to index in probabilities file.
    """
    infile = open(f"d03_classification/viewclasses_{model}.txt")
    views = [i.rstrip() for i in infile.readlines()]

    views_to_indices = {}
    for i, view in enumerate(views):
        # Skip two indices for "study" and "image" in probabilities file.
        views_to_indices[view] = i + 2

    return views_to_indices


def get_viewprob_lists(model, dicomdir_basename):
    """ Get list of values for every instance in study probabilities file.
    """
    viewfile = f"~/data/03_classification/results/{model}_{dicomdir_basename}_probabilities.txt"
    infile = open(viewfile)
    viewprob_lists = [i.rstrip().split("\t") for i in infile.readlines()]

    return viewprob_lists


def get_viewlists(viewprob_lists, views_to_indices, probthresh=0.5):
    """ Get list of instances with A2C and A4C views based on probabilities.
    """
    viewlist_a2c = []
    viewlist_a4c = []

    # Skip header row
    for viewprobs in viewprob_lists[1:]:
        dicomdir = viewprobs[0]
        filename = viewprobs[1]
        if float(viewprobs[views_to_indices["a4c"]]) > probthresh:
            viewlist_a4c.append(filename)
        elif float(viewprobs[views_to_indices["a2c"]]) > probthresh:
            viewlist_a2c.append(filename)

    return viewlist_a2c, viewlist_a4c
