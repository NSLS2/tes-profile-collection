# Suitcase subpackages should follow strict naming and interface conventions.
# The public API must include Serializer and should include export if it is
# intended to be user-facing. They should accept the parameters sketched here,
# but may also accept additional required or optional keyword arguments, as
# needed.
from itertools import zip_longest
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import toml

import event_model
import suitcase.utils

from pprint import pprint
##from ._version import get_versions

##__version__ = get_versions()["version"]
##del get_versions


def export(gen, directory, file_prefix="{uid}-", **kwargs):
    """
    Export a stream of documents to xdi.

    .. note::

        This can alternatively be used to write data to generic buffers rather
        than creating files on disk. See the documentation for the
        ``directory`` parameter below.

    Parameters
    ----------
    gen : generator
        expected to yield ``(name, document)`` pairs

    directory : string, Path or Manager.
        For basic uses, this should be the path to the output directory given
        as a string or Path object. Use an empty string ``''`` to place files
        in the current working directory.

        In advanced applications, this may direct the serialized output to a
        memory buffer, network socket, or other writable buffer. It should be
        an instance of ``suitcase.utils.MemoryBufferManager`` and
        ``suitcase.utils.MultiFileManager`` or any object implementing that
        interface. See the suitcase documentation at
        https://nsls-ii.github.io/suitcase for details.

    file_prefix : str, optional
        The first part of the filename of the generated output files. This
        string may include templates as in ``{proposal_id}-{sample_name}-``,
        which are populated from the RunStart document. The default value is
        ``{uid}-`` which is guaranteed to be present and unique. A more
        descriptive value depends on the application and is therefore left to
        the user.

    **kwargs : kwargs
        Keyword arguments to be passed through to the underlying I/O library.

    Returns
    -------
    artifacts : dict
        dict mapping the 'labels' to lists of file names (or, in general,
        whatever resources are produced by the Manager)

    Examples
    --------

    Generate files with unique-identifier names in the current directory.

    >>> export(gen, '')

    Generate files with more readable metadata in the file names.

    >>> export(gen, '', '{plan_name}-{motors}-')

    Include the experiment's start time formatted as YYYY-MM-DD_HH-MM.

    >>> export(gen, '', '{time:%Y-%m-%d_%H:%M}-')

    Place the files in a different directory, such as on a mounted USB stick.

    >>> export(gen, '/path/to/my_usb_stick')
    """
    with Serializer(directory, file_prefix, **kwargs) as serializer:
        for item in gen:
            serializer(*item)

    return serializer.artifacts


class Serializer(event_model.DocumentRouter):
    """
    Serialize a stream of documents to xdi.

    .. note::

        This can alternatively be used to write data to generic buffers rather
        than creating files on disk. See the documentation for the
        ``directory`` parameter below.

    Parameters
    ----------
    directory : string, Path, or Manager
        For basic uses, this should be the path to the output directory given
        as a string or Path object. Use an empty string ``''`` to place files
        in the current working directory.

        In advanced applications, this may direct the serialized output to a
        memory buffer, network socket, or other writable buffer. It should be
        an instance of ``suitcase.utils.MemoryBufferManager`` and
        ``suitcase.utils.MultiFileManager`` or any object implementing that
        interface. See the suitcase documentation at
        https://nsls-ii.github.io/suitcase for details.

    file_prefix : str, optional
        The first part of the filename of the generated output files. This
        string may include templates as in ``{proposal_id}-{sample_name}-``,
        which are populated from the RunStart document. The default value is
        ``{uid}-`` which is guaranteed to be present and unique. A more
        descriptive value depends on the application and is therefore left to
        the user.

    **kwargs : kwargs
        Keyword arguments to be passed through to the underlying I/O library.

    Attributes
    ----------
    artifacts
        dict mapping the 'labels' to lists of file names (or, in general,
        whatever resources are produced by the Manager)
    """

    def __init__(self, directory, file_prefix="{uid}-", transforms=None, **kwargs):

        self._file_prefix = file_prefix
        if transforms is None:
            self._transforms = {}
        else:
            self._transforms = transforms

        self._kwargs = kwargs
        self._templated_file_prefix = ""  # set when we get a 'start' document
        self._event_descriptor_uids = set()
        self._xdi_file_template = None
        self._column_data = OrderedDict()
        self._header_line_buffer = OrderedDict()
        self.columns = None
        self.export_data_keys = None

        if isinstance(directory, (str, Path)):
            # The user has given us a filepath; they want files.
            # Set up a MultiFileManager for them.
            self._manager = suitcase.utils.MultiFileManager(directory)
        else:
            # The user has given us their own Manager instance. Use that.
            self._manager = directory

        # Finally, we usually need some state related to stashing file
        # handles/buffers. For a Serializer that only needs *one* file
        # this may be:
        #
        # self._output_file = None
        #
        # For a Serializer that writes a separate file per stream:
        #
        # self._files = {}
        self._output_file = None

    @property
    def artifacts(self):
        # The 'artifacts' are the manager's way to exposing to the user a
        # way to get at the resources that were created. For
        # `MultiFileManager`, the artifacts are filenames.  For
        # `MemoryBuffersManager`, the artifacts are the buffer objects
        # themselves. The Serializer, in turn, exposes that to the user here.
        #
        # This must be a property, not a plain attribute, because the
        # manager's `artifacts` attribute is also a property, and we must
        # access it anew each time to be sure to get the latest contents.
        return self._manager.artifacts

    def close(self):
        """
        Close all of the resources (e.g. files) allocated.
        """
        self._manager.close()

    # These methods enable the Serializer to be used as a context manager:
    #
    # with Serializer(...) as serializer:
    #     ...
    #
    # which always calls close() on exit from the with block.

    def __enter__(self):
        return self

    def __exit__(self, *exception_details):
        self.close()

    # Each of the methods below corresponds to a document type. As
    # documents flow in through Serializer.__call__, the DocumentRouter base
    # class will forward them to the method with the name corresponding to
    # the document's type: RunStart documents go to the 'start' method,
    # etc.
    #
    # In each of these methods:
    #
    # - If needed, obtain a new file/buffer from the manager and stash it
    #   on instance state (self._files, etc.) if you will need it again
    #   later. Example:
    #
    #   filename = f'{self._templated_file_prefix}-primary.csv'
    #   file = self._manager.open('stream_data', filename, 'xt')
    #   self._files['primary'] = file
    #
    #   See the manager documentation below for more about the arguments to open().
    #
    # - Write data into the file, usually something like:
    #
    #   content = my_function(doc)
    #   file.write(content)
    #
    #   or
    #
    #   my_function(doc, fil
    def start(self, doc):
        if self._xdi_file_template is not None:
            raise Exception("")

        if "config" in doc["md"]["suitcase-xdi"]:
            self._xdi_file_template = toml.loads(
                doc["md"]["suitcase-xdi"]["config"], _dict=OrderedDict
            )
        elif "config-file-path" in doc["md"]["suitcase-xdi"]:
            self._xdi_file_template = toml.load(
                doc["md"]["suitcase-xdi"]["config-file-path"], _dict=OrderedDict
            )
        else:
            raise Exception(
                "configuration must be specified as a file in md["
                "suitcase-xdi"
                "]["
                "config-file-path"
                "]"
                "or as a string in md["
                "suitcase-xdi"
                "]["
                "config"
                "]"
            )

        """
        Use the configuration information to build an ordered dictionary of header fields, eg
        {
            "XDI":                "# XDI/1.0 Bluesky"
            "Column.1":           "# Column.1 = energy eV",
            "Column.2":           "# Column.2 = mutrans",
            "Column.3":           "# Column.3 = i0",
            "Element.symbol":     None
            "Element.edge":       None
            # Mono.d_spacing = 10.0
            # Facility.name = NSLS-II
            # Beamline.name = BMM
            # Beamline.focusing = parabolic mirror
            # Beamline.harmonic_rejection = detuned
            # Beamline.energy = 1000.000 eV
            # Scan.start_time = 2019-09-18T21:49:43.080123
            # Scan.end_time = None
            # Scan.edge_energy = Scan_edge_energy
        }
        """

        # initialize the column data dictionary to None for each column
        self._initialize_column_data_dict()

        # initialize the header line buffer to "None" for every header line
        ###self._initialize_header_line_buffer(start_doc=doc)

        # extract header information from the start document
        self._update_header_lines_from_doc(doc_name="start", doc=doc)

        # Fill in the file_prefix with the contents of the RunStart document.
        # As in, '{uid}' -> 'c1790369-e4b2-46c7-a294-7abfa239691a'
        # or 'my-data-from-{plan-name}' -> 'my-data-from-scan'
        self._templated_file_prefix = self._file_prefix.format(**doc)
        filename = f"{self._templated_file_prefix}.xdi"
        self._output_file = self._manager.open("stream_data", filename, "xt")

        self.columns = tuple([v for k, v in self._xdi_file_template["columns"].items()])
        if len(self.columns) == 0:
            raise ValueError("found no Columns")

        self.export_data_keys = tuple({c["data_key"] for c in self.columns})

        # write the header information we have now
        # the full header will be written when the stop document arrives
        self._write_header()

    def descriptor(self, doc):
        """
        It is possible to see more than one descriptor. Keep a list of all descriptors with data
        to be exported.

        Parameters
        ----------
        doc : dict
            an event-descriptor document
        """
        descriptor_data_keys = doc["data_keys"]
        print("*************export data keys")
        pprint(self.export_data_keys)
        print("*************descriptor data keys")
        pprint(descriptor_data_keys.keys())
        data_keys_intersection = set(self.export_data_keys).intersection(descriptor_data_keys.keys())
        pprint(data_keys_intersection)

        if len(data_keys_intersection) > 0:
        ##if set(self.export_data_keys).issubset(descriptor_data_keys.keys()):
            self._event_descriptor_uids.add(doc["uid"])
        else:
            ...

        self._update_header_lines_from_doc(doc_name="descriptor", doc=doc)

    def event_page(self, doc):
        # There are other representations of Event data -- 'event' and
        # 'bulk_events' (deprecated). But that does not concern us because
        # DocumentRouter will convert these representations to 'event_page'
        # then route them through here.

        if len(self._event_descriptor_uids) == 0:
            print(
                f"have not seen a descriptor with data keys {self.export_data_keys} yet"
            )
        elif doc["descriptor"] in self._event_descriptor_uids:

            # keep a dict of columns of data like:
            #  {
            #    "energy" : array...
            #    "I0"     : None
            #    "If"     : array...
            #  }
            for column in self.columns:
                data_key = column["data_key"]
                print(column)
                # expect to find an array for the data_key
                if self._column_data[data_key] is None and data_key in doc["data"]:
                    if "transform" in column.keys():
                        print("*************** applying a transform!")
                        transform_key = column["transform"]
                        transform_function = self._transforms[transform_key]
                        self._column_data[data_key] = transform_function(doc)
                    else:
                        event_data = doc["data"][data_key][0]  # TODO: why is [0] needed?
                        print(f"found data for data key {data_key}")
                        pprint(event_data)
                        if isinstance(event_data, np.ndarray):
                            self._column_data[data_key] = event_data
                        else:
                            self._column_data[data_key] = (event_data, )

        else:
            # this event has no data to export
            pass

    def stop(self, doc):
        self._update_header_lines_from_doc(doc_name="stop", doc=doc)
        self._manager.close()
        for artifact_label, artifacts in self._manager.artifacts.items():
            for artifact in artifacts:
                print("finishing artifact {}".format(artifact))
                temp_artifact_path = artifact.with_suffix(".updating")
                print("creating {}".format(temp_artifact_path))
                with artifact.open() as a, temp_artifact_path.open("wt") as t:
                    # write a fresh header
                    self._write_header(output_file=t)
                    # write the data
                    for row_data in zip_longest(
                        *self._column_data.values(),
                        fillvalue="NA"
                    ):
                        t.write("\t".join((str(d) for d in row_data)))
                        t.write("\n")

                artifact.unlink()
                temp_artifact_path.rename(artifact)

    def _initialize_column_data_dict(self):
        for xdi_key, xdi_value in self._xdi_file_template["columns"].items():
            self._column_data[xdi_value["data_key"]] = None

    def _update_header_lines_from_doc(self, doc_name, doc):
        """
        Initialize or update self._header_line_buffer using information in the
        specified document.

        The first time through self._header_line_buffer is empty. At this point
        all the XDI header keys should be inserted with value None.

        Subsequent calls will insert values (full XDI header rows) only for
        entries with value None. Entries with a header row value will not be
        updated.

        {}

        becomes

        {
          "XDI": "# XDI/1.0 Bluesky",
          "Column.1": None,
          "Element.symbol": None,
          ...
        }

        becomes

        {
          "XDI": "# XDI/1.0 Bluesky",
          "Column.1": "energy eV",
          "Element.symbol": "K",
          ...
        }

        :param doc_name:
        :param doc:
        :return:
        """
        def _get_empty_header_lines(config_section_name):
            """
            Return entries from self._xdi_file_template to be used to generate header rows
            using information from a document. Each (key, value) pair returned meets
            two criteria:
              1. 'key' is in both self._xdi_file_template and self._header_line_buffer
              2. self._header_line_buffer[key] is None (meaning this header row has not been generated yet)

            Parameters
            ----------
            config_section_name: str
                one of the named sections of the configuration dictionary:
                "versions", "columns", "required_headers", or "optional_headers"

            Return
            ------
            list of (key, value) entries from self._xdi_file_template meeting the criteria specified above
            """
            for _header_label in self._xdi_file_template[config_section_name].keys():
                # initialize self._header_line_buffer[k] if it does not exist
                if _header_label not in self._header_line_buffer:
                    self._header_line_buffer[_header_label] = None
                else:
                    pass
                # yield (header_label, self._xdi_file_template[header_label]) if
                # the corresponding header line has not been generated
                if self._header_line_buffer[_header_label] is None:
                    yield _header_label, self._xdi_file_template[config_section_name][_header_label]
                else:
                    pass

        for header_label, version in _get_empty_header_lines("versions"):
            self._header_line_buffer[header_label] = version

        for header_label, column_line_template in _get_empty_header_lines("columns"):
            header_value = column_line_template["column_label"].format(**doc)
            if "units" in column_line_template:
                self._header_line_buffer[header_label] = (
                    f"# {header_label} = {header_value} " + column_line_template["units"]
                )
            else:
                self._header_line_buffer[header_label] = f"# {header_label} = {header_value}"

        for header_label, column_line_template in _get_empty_header_lines("required_headers"):
            header_value = column_line_template["data"].format(**doc)
            self._header_line_buffer[header_label] = f"# {header_label} = {header_value}"

        for header_label, column_line_template in _get_empty_header_lines("optional_headers"):
            if header_label == "Scan.start_time" and doc_name == "start":
                header_value = datetime.fromtimestamp(doc["time"]).isoformat()
            elif header_label == "Scan.end_time" and doc_name == "stop":
                header_value = datetime.fromtimestamp(doc["time"]).isoformat()
            else:
                header_value = column_line_template["data"].format(**doc)
            self._header_line_buffer[header_label] = f"{header_label} = {header_value}"

    def _write_header(self, output_file=None):
        """Write all header information, "None" for missing information.

        """
        if output_file is None:
            output_file = self._output_file

        for header_field, header_value in self._header_line_buffer.items():
            output_file.write(header_value)
            output_file.write("\n")

        output_file.write("#----\n")
        output_file.write(
            "# {}\n".format("\t".join([c["column_label"] for c in self.columns]))
        )

