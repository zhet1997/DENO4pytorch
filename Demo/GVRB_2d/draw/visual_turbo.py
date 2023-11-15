import os

from Utilizes.visual_data import MatplotlibVision

from Tools.post_process.post_data import Post_2d


class MatplotlibTurbo(object):
    def __init__(self,log_dir=None,input_name=('Z', 'R'), field_name=('n')):
        if log_dir is None:
            log_dir = os.getcwd()# save in the current path as default
        # establish the plot object
        self.Vision = MatplotlibVision(log_dir, input_name=input_name, field_name=field_name)
        self.parameterList = None
        self.value = dict()
        self.flow = dict()
        self.post = dict()

    def get_turbo(self, post, id):
        assert not isinstance(post,Post_2d)
        self.post[id] = post
        self.value[id] = dict()
        self.flow[id] = dict()

    def calculate_S1_contour(self, parameterList=None, idList=None):
        if not isinstance(parameterList,list):
            parameterList = [parameterList]
        if parameterList is None:
            assert self.parameterList is None, "Please enter the parameters your want"
            parameterList = self.parameterList
        if idList is None:
            idList = self.post.keys()

        for id in idList:
            for parameter in parameterList:
                self.value[id][parameter] = getattr(self.post[id], parameter)

        self.parameterList = parameterList

    def calculate_flow_curve(self, parameterList=None, idList=None):
        if not isinstance(parameterList, list):
            parameterList = [parameterList]
        if parameterList is None:
            assert self.parameterList is None, "Please enter the parameters your want"
            parameterList = self.parameterList
        if idList is None:
            idList = self.post.keys()

        for id in idList:
            for parameter in parameterList:
                self.flow[id][parameter] = self.post[id].field_density_average(parameter, location="whole")

        self.parameterList = parameterList

    def get_span_value(self, parameterList=None, idList=None, span_idx=-1):
        if not isinstance(parameterList,list):
            parameterList = [parameterList]
        if parameterList is None:
            assert self.parameterList is None, "Please enter the parameters your want"
            parameterList = self.parameterList
        if idList is None:
            idList = self.post.keys()

        return 0

