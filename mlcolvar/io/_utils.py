import tempfile
import urllib.request
import os


def _download_temp_file(file_url: str,
                        delete_download: bool = True,
                        append_suffix: bool = False,
                        return_name: bool = False
                       ):
    if delete_download:
        if append_suffix:
            temp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_url)[1].lower() )
        else:
            temp = tempfile.NamedTemporaryFile()
        file_name = temp.name   
    else:
        temp = None
        file_name = "tmp_" + file_url.split("/")[-1]
    urllib.request.urlretrieve(file_url, file_name)
    
    return temp if not return_name else temp, file_name