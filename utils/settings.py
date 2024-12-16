

class Settings:

    # General

    class General:

        # For different image based models with fixed inputs, 224, 224 for Resnet50
        def_compress_width: int = 224
        def_compress_height: int = 224

    # INBreast

    class INBreast:

        unzip_path: str = 'INBreast'

    #CBIS-DDSM

    class CBIS_DDSM:

        unzip_path: str = 'CBIS-DDSM'
