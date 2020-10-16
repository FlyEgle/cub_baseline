"""
-*-coding:utf-8
config for different network with diff input size
"""

class ModelSize:
    def __init__(self, net_name):
        self.net_name = net_name

    def _regnet_320(self):
        return {"resize": 384, "input": 336 }

    def _resnet50_448(self):
        return {"resize": 512, "input": 448}

    def _resnet50_224(self):
        return {"resize": 256, "input": 224}

    def _efnet_b0_224(self):
        return {"resize": 256, "input": 224}

    def _efnet_b1_240(self):
        return {"resize": 272, "input": 240}

    def _efnet_b2_260(self):
        return {"resize": 292, "input": 260}

    def _efnet_b3_300(self):
        return {"resize": 352, "input": 300}

    def _efnet_b4_380(self):
        return {"resize": 416, "input": 380}

    def _efnet_b5_456(self):
        return {"resize": 512, "input": 456}

    def _efnet_b6_528(self):
        return {"resize": 600, "input": 528}

    def _efnet_b7_600(self):
        return {"resize": 640, "input": 600}

    def imagesize_choice(self):
        if self.net_name == "resnet50_448":
            return self._resnet50_448()
        elif self.net_name == "resnet50":
            return self._resnet50_224()
        elif self.net_name == "efnet-b0":
            return self._efnet_b0_224()
        elif self.net_name == "efnet-b1":
            return self._efnet_b1_240()
        elif self.net_name == "efnet-b2":
            return self._efnet_b2_260()
        elif self.net_name == "efnet-b3":
            return self._efnet_b3_300()
        elif self.net_name == "efnet-b4":
            return self._efnet_b4_380()
        elif self.net_name == "efnet-b5":
            return self._efnet_b5_456()
        elif self.net_name == "efnet-b6":
            return self._efnet_b6_528()
        elif self.net_name == "efnet-b7":
            return self._efnet_b7_600()
        elif self.net_name == "regnet_320":
            return self._regnet_320()