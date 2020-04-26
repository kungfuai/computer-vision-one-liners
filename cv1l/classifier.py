import click
from glob import glob
import numpy as np
from os import path
from tensorflow.keras import (
    applications,
    layers,
    models,
    metrics,
    callbacks,
    optimizers,
    losses,
    utils,
    preprocessing,
)


# TODO: move this class into a module.
class DataSourceValidator:
    def run(self):
        pass


def validate_input_directory():
    """
    Make sure the structure of the input directory is supported.
    """
    pass


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("--output_dir", "-o", default="output", help="Output directory.")
def main(input_dir, output_dir):
    class_names = [path.basename(s) for s in glob(input_dir + "/*/")]
    n_classes = len(class_names)
    # image_gen = preprocessing.image.ImageDataGenerator(
    #     rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    # )
    image_gen = preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
    train_gen = image_gen.flow_from_directory(
        input_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode="categorical",  # "binary"
        color_mode="rgb",
    )
    base_model = applications.mobilenet_v2.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet", pooling="max"
    )
    for layer in base_model.layers:
        layer.trainable = False
    y = base_model.output
    y = layers.Dense(n_classes, activation="softmax")(y)
    model = models.Model(inputs=base_model.inputs, outputs=y)
    lr = 0.05
    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", metrics.AUC()],
    )
    # print(model.summary())
    model.fit(
        train_gen,
        validation_data=train_gen,
        epochs=20,
        callbacks=[
            callbacks.ModelCheckpoint(output_dir, save_best_only=True),
            callbacks.EarlyStopping(patience=2),
            callbacks.LearningRateScheduler(
                lambda epoch: lr * np.exp(-0.1 * (epoch - 1))
            ),
            callbacks.CSVLogger(path.join(output_dir, "metrics.csv")),
        ],
    )


if __name__ == "__main__":
    main()
