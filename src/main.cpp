#include <iostream>
#include <gst/gst.h>

int main(int argc, char *argv[]) {
    // gstreamer initialization
    gst_init(&argc, &argv);

    // building pipeline
    auto pipeline = gst_parse_launch(
        "v4l2src ! jpegdec ! videoconvert ! video/x-raw, format=(string)RGB ! cudafilter ! videoconvert ! fpsdisplaysink",
        nullptr);

    // start playing
    auto ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr ("Unable to set the pipeline to the playing state.\n");
        gst_object_unref (pipeline);
        return -1;
    }

    //wait until error or EOS ( End Of Stream )
    auto bus = gst_element_get_bus(pipeline);
    auto msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                     static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    /* See next tutorial for proper error message handling/parsing */
    if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR) {
      g_error ("An error occurred! Re-run with the GST_DEBUG=*:WARN environment "
          "variable set for more details.");
    }

    /* Free resources */
    gst_message_unref (msg);
    gst_object_unref (bus);
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (pipeline);

    return 0;
}