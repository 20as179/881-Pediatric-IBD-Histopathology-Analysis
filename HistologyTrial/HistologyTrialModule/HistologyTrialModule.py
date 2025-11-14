import logging
import os
from typing import Annotated, Optional

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



#import matplotlib.pyplot as plt
#from skimage.color import rgb2hed, hed2rgb


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
        self.parent.contributors = ["Laavanya and Ally"]  # TODO: replace with "Firstname Lastname (Organization)"
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
        self.logic = HistologyTrialModuleLogic()

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
            hema_stack_node, eosin_stack_node = self.logic.process(self.ui.inputSelector.currentNode(),
                               self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value,
                               self.ui.invertOutputCheckBox.checked)

            green = slicer.app.layoutManager().sliceWidget("Green")
            green_comp = green.mrmlSliceCompositeNode()
            green_comp.SetBackgroundVolumeID(hema_stack_node.GetID())
            green.fitSliceToBackground()

            yellow = slicer.app.layoutManager().sliceWidget("Yellow")
            yellow_comp = yellow.mrmlSliceCompositeNode()
            yellow_comp.SetBackgroundVolumeID(eosin_stack_node.GetID())
            yellow.fitSliceToBackground()


            volumeNode = self.ui.inputSelector.currentNode()
            if not volumeNode:
                logging.warning("No volume selected")
                return

            #self.logic.showHematoxylinVolume(volumeNode)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(),
                                   self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value,
                                   not self.ui.invertOutputCheckBox.checked, showResult=False)




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

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)


    def getSegmentedHistologyImages(self, vtkMRMLVectorVolumeNode):

        #volume_node = slicer.mrmlScene.GetFirstNodeByClass(vtkMRMLVectorVolumeNode)

        #array = slicer.util.arrayFromVolume(volume_node)
        #print(array)

        volume_array = slicer.util.arrayFromVolume(vtkMRMLVectorVolumeNode)

        #volume = sitk.ReadImage(input_volume)
        #volume_array = sitk.GetArrayFromImage(volume)
        #print(f"Volume shape: {volume_array.shape}")

        #volume_array = volume_array.transpose((1, 0, 2, 3))
        volume_array = volume_array.astype(np.float32) / 255.0

        hematoxylin_slices = []
        eosin_slices = []
        for i in range(volume_array.shape[0]):
            # convert to RGB
            image_rgb = cv2.cvtColor(volume_array[i, :, :], cv2.COLOR_BGR2RGB)
            # convert to HED
            image_hed = rgb2hed(image_rgb)

            null = np.zeros_like(image_hed[:, :, 0])
            hematoxylin = hed2rgb(np.stack((image_hed[:, :, 0], null, null), axis=-1))
            eosin = hed2rgb(np.stack((null, image_hed[:, :, 0], null), axis=-1))

            # Normalize to [0,1] range
            '''hematoxylin = image_hed[:, :, 0]
            eosin = image_hed[:, :, 1]

            hematoxylin = (hematoxylin - hematoxylin.min()) / (hematoxylin.max() - hematoxylin.min())
            eosin = (eosin - eosin.min()) / (eosin.max() - eosin.min())
            eosin_slices.append(eosin)'''

            hematoxylin_slices.append(hematoxylin)
            eosin_slices.append(eosin)

        # stack slices and convert to volume
        hematoxylin_stack = (np.stack(hematoxylin_slices, axis=0))#.astype(np.float32)
        hema_stack_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLVectorVolumeNode', "Hematoxylin Staining")
        slicer.util.updateVolumeFromArray(hema_stack_node, hematoxylin_stack)

        eosin_stack = (np.stack(eosin_slices, axis=0))#.astype(np.float32)
        eosin_stack_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLVectorVolumeNode', "Eosin Staining")
        slicer.util.updateVolumeFromArray(eosin_stack_node, eosin_stack)

        return hema_stack_node, eosin_stack_node


        #hematoxylin_volume = sitk.GetImageFromArray(hematoxylin_stack, isVector=False)
        #print(hematoxylin_volume.shape)
        '''hematoxylin_volume_path = os.path.join(output_volume_dir, f"{os.path.basename(input_dir)}_hematoxylin_volume.nrrd")
        sitk.WriteImage(hematoxylin_volume, hematoxylin_volume_path)
        print(f"Saved volume as: {hematoxylin_volume_path}")'''

        eosin_stack = np.stack(eosin_slices, axis=0)
        eosin_volume = sitk.GetImageFromArray(eosin_stack, isVector=False)
        #print(eosin_volume.shape)
        '''eosin_volume_path = os.path.join(output_volume_dir, f"{os.path.basename(input_dir)}_eosin_volume.nrrd")
        sitk.WriteImage(eosin_volume, eosin_volume_path)
        print(f"Saved volume as: {eosin_volume_path}")'''


        # load images
        #resized_image = cv2.imread(input_path)

        # convert to HED and separate channels
        '''image_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)  # convert to RGB
        image_hed = rgb2hed(image_rgb)  # Convert RGB to HED

        hematoxylin = image_hed[:, :, 0]

        null = np.zeros_like(image_hed[:, :, 0])   https://scikit-image.org/docs/0.25.x/auto_examples/color_exposure/plot_ihc_color_separation.html
        #fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
        #ax = axes.ravel()
        ihc_h = hed2rgb(np.stack((image_hed[:, :, 0], null, null), axis=-1))

        ihchNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")

        # Set the image data
        slicer.util.updateVolumeFromArray(ihchNode, hematoxylin)

        # Optionally, adjust display properties
        displayNode = ihchNode.GetDisplayNode()
        displayNode.SetAndObserveColorNodeID(slicer.app.getColorNodes().GetID())'''


        '''ax[1].imshow(ihc_h)
        #ax[1].set_title("Hematoxylin")
        #print("test2")'''

        #return hematoxylin
    '''
        ihc_e = hed2rgb(np.stack((null, image_hed[:, :, 1], null), axis=-1))
        ax[2].imshow(ihc_e)
        #ax[2].set_title("Eosin")
        #print("test")

        ihc_d = hed2rgb(np.stack((null, null, image_hed[:, :, 2]), axis=-1))
        ax[3].imshow(ihc_d)'''
        #ax[3].set_title("DAB")
        #print("test3")


    '''def showHematoxylinVolume(self, hema_stack_node):
        green = slicer.app.layoutManager().sliceWidget("Green")
        green_comp = green.mrmlSliceCompositeNode()
        green_comp.SetBackgroundVolumeID(hema_stack_node.GetID())'''
        #slicer.app.applicationLogic().GetSliceLogic("Green").FitSliceToVolume(hema_stack_node)



    def process(self,
                inputVolume: vtkMRMLVectorVolumeNode,
                outputVolume: vtkMRMLVectorVolumeNode,
                imageThreshold: float,
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
        hema_stack_node, eosin_stack_node = self.getSegmentedHistologyImages(inputVolume)

        '''hema_stack_node.CreateDefaultDisplayNodes()
        slicer.util.setSliceViewerLayers(background=hema_stack_node)
        slicer.util.resetSliceViews()'''

        return hema_stack_node, eosin_stack_node

        #pass WHAT SHE ADDED


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

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
