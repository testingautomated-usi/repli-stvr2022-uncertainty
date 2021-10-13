import tensorflow as tf
import uncertainty_wizard


def limit_gpu_use(gpu_index: int = 0, memory_limit_kb: int = 3072):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_index],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_kb)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)


def disable_gpu_use():
    tf.config.experimental.set_visible_devices([], 'GPU')


def use_only_gpu(gpu_index: int):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices([gpus[gpu_index]], 'GPU')
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    else:
        print("No GPUs available")

def dynamic_growth():
    uncertainty_wizard.models._lazy_ensemble