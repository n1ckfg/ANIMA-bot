# FilmStripAni Animation Applet v1.0

The purpose of this applet is to take a "Film Strip" (both vertical and

horizontal layouts) type image and give it animation like a film projector.

The animation can be stopped and re-started at any time by clicking on the

applet/image.

The animation is completely configurable. All of the valid parameters are

described within this document.

The image is double buffered for fluidity.  I have noticed, on occasion, when

the applet size is larger that the frame size, a remnant of the parts of the

image that are clipped and are not in view appear during scrolling, but

immediately go away when the next thread catches up or when the thread is

re-started.  I have looked into this and I am not sure what is causing this

and whether or not there is anything that I can do about it.  One definite

way to avoid this is to turn on Debugging and find the exact frame size, and

then reduce your applet size to the frame size.

To add this applet to your page, simply add the following Java applet

tag to your HTML code, and modify the parameter values to meet your own

individual needs. Since a Java applet is just another program running

on your computer, it has the ability to take input from a user. This program

gets its input from parameters sent to it from the HTML tag. All parameters

are optional except for the HEIGHT and WIDTH which are

a part of all Java APPLET tags. The different parameters will be

described below.

This following sample HTML code is typical of the parameters needed to get an

image animated.  Again, the applet can be configured to meet the needs of your

particular image.

```
<applet code="FilmStripAni"
            codebase="http://www.netobjective.com/java/classes"
            width=100 height=400>
                <param name="IMAGE" value="putt.jpg">
                <param name="BORDERCOLOR" value="ffffff">
                <param name="FRAMES" value="21">
                <param name="FRAMEGAP" value="3">
                <param name="FRAMESPEED" value="200">
        </applet>
```

The values of the HEIGHT and WIDTH values can be adjusted

accordingly.  If the applets dimensions are larger than the size of a frame,

the space around the frame will be filled in with the color passed to the

BORDERCOLOR parameter (or white by default).  The above tag will provide the

following applet:

##### (Click on the applet Stop and Re-Start it.)

##### [Click me to see the Full Image.](putt.jpg)

##### [Created by DNDesign .](http://www.netobjective.com/dndesign)

Parameters are passed to Java by using the PARAM tag and the

parameter name and corresponding value. The PARAM tag needs to

be in between the beginning &lt;APPLET&gt; and ending &lt;/APPLET&gt; tags. This applet can be customized by using the

following optional parameters:

### Image (Image Information) Parameters:

- **IMAGE** The Name of the image that is to be animated. The image can be any of the typical WWW image formats. I have tried and tested an image in both JPEG and GIF formats. The image is being loaded via the Media Tracker and therefore, you will be notified of its progress. If there is a loading error, a message will appear, but you will have to reference the Java Console for any additional information. **VERTICAL** Is the image layout Vertically (top to bottom)? The default layout is HORIZONTAL (left to right). If you want to aimation to happen in reverse, invert the image.
    - DEFAULT: "image.jpg"
    - EXAMPLE: &lt;param name="IMAGE" value="sprockets.gif"&gt;

### Frame (Information about each of the Frames) Parameters:

- **FRAMES** The total number of individual frames in the image. In most cases, the more frames the better, but like most of you, I am no artist. **FRAMEGAP** The space, if any, between the frames. If you notice that the image moves to the left during animation, chances are pretty good that there is a small gap between the frames. **FRAMESPEED** The Speed at which the Animation thread refreshes itself. Some animations look better faster. It can hide that fact that the image does not contain a lot of detail.
    - DEFAULT: "10"
    - EXAMPLE: &lt;param name="FRAMES" value="4"&gt;

### Other (Non-Frame, Animation or Image Related) Parameters:

- **BORDERCOLOR** If the applet size is larger than the frame size, you can pass a hexadecimal color and all of the space around the frame will be filled with the color. It may difficult to look at an image and determine the frame size (pass the DEBUG flag to get the exact frame size) for a perfect fit. So, a border color can be used to replace the applet default background of grey. **DEBUG** If the DEBUG parameter is passed, irregardless of the value, debugging output will be sent to the Java Console.
    - DEFAULT: "white" or "ffffff;"
    - EXAMPLE: &lt;param name="BORDERCOLOR" value="ffff00;"&gt;

Using the EXAMPLES shown above we yield the following HTML:

```
<applet code="FilmStripAni"
		    codebase="http://www.netobjective.com/java/classes"
		    height=100 width=100>
            <param name="IMAGE" value="sprockets.gif">
            <param name="VERTICAL" value="">
            <param name="BORDERCOLOR" value="ffff00">
            <param name="FRAMES" value="4">
            <param name="FRAMEGAP" value="0">
            <param name="FRAMESPEED" value="200">
        </applet;>
```

Implementing the above listed applet tags (from all of the EXAMPLES), with

a table placed around it for flash and dazzle, produces the following:

##### (Click on the applet Stop and Re-Start it.)

##### [Click me to see the Full Image.](sprockets.gif)

## To Do List

- So far, nothing. I need feedback!

Please send any and all comments, suggestions, or questions to [FilmStripAni Info/Help](mailto:wjdavis@netobjective.com)

Number of Accesses: Back to [NetObjective's](http://www.netobjective.com/) Home Page Back to [NetObjective's Java](http://www.netobjective.com/java) Page Copyright 1996 NetObjective

<!-- 🖼️❌ Image not available. Please use `PdfPipelineOptions(generate_picture_images=True)` -->