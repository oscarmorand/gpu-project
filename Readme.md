

## Mise en route

1. Installer gstreamer (voir le fichier nix-shell associé)
2. Builder le code d'exemple avec cmake (le filtre se trouve dans le dossier ``./build``)
3. Télécharger la vidéo d'exemple ``https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm``
4. Exporter le chemin du filtre dans la variable d'environnement ``GST_PLUGIN_PATH``
5. Ajouter un symlink vers le plugin C++ ou sa version CUDA
6. Lancer l'application du filter sur la vidéo et l'enregistrer en mp4 *dans votre afs*
7. En local, visualiser la vidéo avec *vlc*


```sh
nix-shell                                            # 1
cmake -S . -B build --preset release -D USE_CUDA=ON  # 2 (ou debug)
cmake --build build                                  # 2


wget https://gstreamer.freedesktop.org/media/sintel_trailer-480p.webm # 3
export GST_PLUGIN_PATH=$(pwd)                                         # 4
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so             # 5
ln -s ./build/libgstcudafilter-cu.so libgstcudafilter.so 


gst-launch-1.0 uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4 #5

gst-launch-1.0 uridecodebin uri=file://$(pwd)/subject/camera.mp4 ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter th_low=4 th_high=30 ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4
```

## Code

Les seuls fichiers à modifier sont normalement ``filter_impl.cu`` (version cuda) et ``filter_impl.cpp`` (version cpp). Pour basculer entre l'utilisation du filter en C++ et du filtre en CUDA, changer le lien symbolique vers le bon ``.so``.


## Uiliser *gstreamer*

### Flux depuis la webcam -> display

Si vous avez une webcam, vous pouvez lancer gstreamer pour appliquer le filter en live et afficher son FPS.

```sh
gst-launch-1.0 -e -v v4l2src ! jpegdec ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

### Flux depuis une vidéo locale -> display

Même chose pour une vidéo en locale.

```sh
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

## Flux depuis une vidéo locale -> vidéo locale

Pour sauvegarder le résulat de l'application de votre filtre.

```sh
gst-launch-1.0 uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=video.mp4
```


## Bench FPS du traitement d'une vidéo

Enfin pour bencher la vitesse de votre filtre. Regarder la sortie de la console pour voir les fps. 

```sh
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/sintel_trailer-480p.webm !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false
```