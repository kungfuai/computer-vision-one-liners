# Computer vision one line commands

Less is more.

## Setup

- Install `docker` on your machine.
- Clone this repo to your machine: `git clone https://github.com/kungfuai/computer-vision-one-liners`.
- Build it: `./bin/build`.

## Here we go

### Train a binary image classifier, e.g. car in photo vs. not.

```
./bin/train <input_directory> [-o <output_directory>]
```

Your `input_directory` can look like this:

```
my_input_directory/
    car/
      1.jpg
      2.jpg
    no_car/
      1000.jpg
      1001.jpg
```

### Train a multi-class image classifier, e.g. cat vs. dog vs. horse.

```
./bin/train <input_directory> [-o <output_directory>]
```

Your `input_directory` can look like this:

```
my_input_directory/
    cat/
      1.jpg
      2.jpg
    dog/
      1000.jpg
      1001.jpg
    horse/
      2000.jpg
      2001.jpg
```

### Train a multi-label binary image classifier, i.e. photo2tags.

```
./bin/train <input_directory> [-o <output_directory>]
```

Your `input_directory` can look like this:

```
my_input_directory/
    labels.txt
    images/
      1.jpg
      2.jpg
      Photo 3.jpg
      a_sub_folder/
        4.jpg
      ...
```

The `labels.txt` file has the labels/tags per image. Each line starts with the relative path of an image, then its tags separated by a space:

```
images/1.jpg cat blue_sky my-favorate
"images/Photo 3.jpg" cat
imags/a_sub_folder/4.jpg dog indoor
```