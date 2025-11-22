import logging
import os
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLVectorVolumeNode


try:
    import numpy as np
except ModuleNotFoundError:
    slicer.util.pip_install("numpy")
    import numpy as np


try:
    import cv2
except ModuleNotFoundError:
    slicer.util.pip_install("opencv-python")
    import cv2

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    slicer.util.pip_install("matplotlib.pyplot")
    import matplotlib.pyplot as plt

try:
    import SimpleITK as sitk
except ModuleNotFoundError:
    slicer.util.pip_install("SimpleITK")
    import SimpleITK as sitk

try:
    import skimage.color
except ModuleNotFoundError:
    slicer.util.pip_install("skimage.color")
    import skimage.color

from skimage.color import rgb2hed, hed2rgb

try:
    import tensorflow
except ModuleNotFoundError:
    slicer.util.pip_install("tensorflow")
    import tensorflow

try:
    import keras
except ModuleNotFoundError:
    slicer.util.pip_install("keras")
    import keras

from keras.models import load_model


#
# HistologyTrialModule
#


class HistologyTrialModule(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("HistologyTrialModule")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Laavanya Joshi and Ally Shi"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#HistologyTrialModule">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # HistologyTrialModule1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="HistologyTrialModule",
        sampleName="HistologyTrialModule1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "HistologyTrialModule1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="HistologyTrialModule1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="HistologyTrialModule1",
    )

    # HistologyTrialModule2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="HistologyTrialModule",
        sampleName="HistologyTrialModule2",
        thumbnailFileName=os.path.join(iconsPath, "HistologyTrialModule2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="HistologyTrialModule2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="HistologyTrialModule2",
    )


#
# HistologyTrialModuleWidget
#


class HistologyTrialModuleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/HistologyTrialModule.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        #self.logic = HistologyTrialModuleLogic()

        model_path = os.path.join(self.resourcePath("VGG16_histology_classification.keras"))
        #full_path = self.resourcePath(r"Resources/VGG16_histology_classification.keras")
        self.logic = HistologyTrialModuleLogic(model_path)

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)


    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        pass#self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        pass#self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            pass#self.initializeParameterNode()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume:# and self._vtkMRMLVectorVolumeNode
            self.ui.applyButton.toolTip = _("apply button test")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("button test failed")
            self.ui.applyButton.enabled = False


    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            hema_stack_node, eosin_stack_node, diagnosis = self.logic.process(self.ui.inputSelector.currentNode())

            green = slicer.app.layoutManager().sliceWidget("Green")
            green_comp = green.mrmlSliceCompositeNode()
            green_comp.SetBackgroundVolumeID(hema_stack_node.GetID())
            green.fitSliceToBackground()

            yellow = slicer.app.layoutManager().sliceWidget("Yellow")
            yellow_comp = yellow.mrmlSliceCompositeNode()
            yellow_comp.SetBackgroundVolumeID(eosin_stack_node.GetID())
            yellow.fitSliceToBackground()

            self.ui.diagnosisLabel.text = str(self.logic.diagnosis)

            volumeNode = self.ui.inputSelector.currentNode()
            if not volumeNode:
                logging.warning("No volume selected")
                return




#
# HistologyTrialModuleLogic
#


class HistologyTrialModuleLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, model_path) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

        super().__init__()
        self.model_path = model_path
        self.model = tensorflow.keras.models.load_model(self.model_path)


    def getSegmentedHistologyImages(self, vtkMRMLVectorVolumeNode):
        '''
        This function takes in the volumetric stack of images, separated the hematoxylin and eosin channels, creating new volumetric images, and displays them. It also runs the CNN to generate a diagnosis for the given patient.
        :param vtkMRMLVectorVolumeNode:
        :return: hematoxylin volumetric data, eosin volumetric data, predictive diagnosis
        '''

        volume_array = slicer.util.arrayFromVolume(vtkMRMLVectorVolumeNode)

        volume_array = volume_array.astype(np.float32) / 255.0

        hematoxylin_slices = []
        eosin_slices = []
        for i in range(volume_array.shape[0]):
            # convert to RGB
            image_rgb = cv2.cvtColor(volume_array[i, :, :], cv2.COLOR_BGR2RGB)
            # convert to HED
            image_hed = rgb2hed(image_rgb)

            # Perform the colour separation for each of the dyes
            null = np.zeros_like(image_hed[:, :, 0])
            hematoxylin = hed2rgb(np.stack((image_hed[:, :, 0], null, null), axis=-1))
            eosin = hed2rgb(np.stack((null, image_hed[:, :, 1], null), axis=-1))

            # Rotate the image to view appropriately and in-line with the original image
            hematoxylin = np.rot90(hematoxylin, k = 2)
            eosin = np.rot90(eosin, k = 2)

            # Add the image to the new volumetric image, one for each stain
            hematoxylin_slices.append(hematoxylin)
            eosin_slices.append(eosin)

        # set up a cumulative score, that will contain the diagnosis from each image in the hematoxylin volumetric data
        model_results = []

        for i in range(len(hematoxylin_slices)):

            # resize images to an appropriate size for VGG16
            resized_hematoxylin = cv2.resize(hematoxylin_slices[i], (224, 224), interpolation=cv2.INTER_AREA)
            resized_hematoxylin = np.expand_dims(resized_hematoxylin, axis=0)

            # run each image through the model
            prediction = self.model.predict(resized_hematoxylin)
            model_results.append(prediction)

        # Get an average score from entire volumetric data, and diagnose the patient based on the mean
        mean_prediction = np.mean(model_results)
        if mean_prediction < 0.5:
            diagnosis = "Ulcerative Colitis"
        else:
            diagnosis = "Crohn's Disease"


        # stack slices and convert to volume
        hematoxylin_stack = (np.stack(hematoxylin_slices, axis=0))#.astype(np.float32)
        hema_stack_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLVectorVolumeNode', f"Hematoxylin Staining")
        slicer.util.updateVolumeFromArray(hema_stack_node, hematoxylin_stack)

        eosin_stack = (np.stack(eosin_slices, axis=0))#.astype(np.float32)
        eosin_stack_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLVectorVolumeNode', "Eosin Staining")
        slicer.util.updateVolumeFromArray(eosin_stack_node, eosin_stack)


        return hema_stack_node, eosin_stack_node, diagnosis



    def process(self,
                inputVolume: vtkMRMLVectorVolumeNode,
                #outputVolume: vtkMRMLVectorVolumeNode,
                #imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """
        hema_stack_node, eosin_stack_node, self.diagnosis = self.getSegmentedHistologyImages(inputVolume)

        return hema_stack_node, eosin_stack_node, self.diagnosis


#
# HistologyTrialModuleTest
#


class HistologyTrialModuleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_HistologyTrialModule1()

    def test_HistologyTrialModule1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("HistologyTrialModule1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = HistologyTrialModuleLogic()


