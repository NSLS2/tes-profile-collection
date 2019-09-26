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


def export(
    gen,
    directory,
    file_prefix="{scan_title}-",
    xdi_file_template=None,
    transforms=None,
    **kwargs,
):
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
    with Serializer(
        directory,
        file_prefix,
        xdi_file_template=xdi_file_template,
        transforms=transforms,
        **kwargs,
    ) as serializer:
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

    def __init__(self, directory, file_prefix="{scan_title}-", **kwargs):

        if isinstance(directory, (str, Path)):
            # The user has given us a filepath; they want files.
            # Set up a MultiFileManager for them.
            self._manager = suitcase.utils.MultiFileManager(directory)
        else:
            # The user has given us their own Manager instance. Use that.
            self._manager = directory

        self._file_prefix = file_prefix

        if "xdi_file_template" not in kwargs or kwargs["xdi_file_template"] is None:
            self._xdi_file_template = None
        else:
            self._xdi_file_template = toml.loads(kwargs["xdi_file_template"])

        if "transforms" not in kwargs or kwargs["transforms"] is None:
            self._transforms = dict()
        else:
            self._transforms = kwargs["transforms"]

        self._kwargs = kwargs  # needed?
        self._templated_file_prefix = None  # set when we get a 'start' document

        self._uid_to_descriptor = dict()

        # when writing files header information will be taken from self._header_line_buffer
        # and self._event_page_header_line_buffer. Header lines in the former will be the
        # same across all output files. Header lines in the latter will vary across output
        # files.
        self._header_line_buffer = dict()
        # self._event_page_header_line_buffer = None

        # use list self._row_end_docs to discriminate a "begin row" event_page from
        # an "end row" event page
        self._row_end_docs = list()

        # use seq_num from the primary stream event_pages to determine the scan number
        self._scan_number = None

        # the column data for each output file will be found in event_pages
        # but that file can not be written until a later event_page has been
        # handled, so self._column_data will hold it until the serializer
        # is ready to write
        self._column_data = dict()

        self.columns = None
        self.export_data_keys = None

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

        if self._xdi_file_template is None:
            if "config" in doc["md"]["suitcase-xdi"]:
                print(doc["md"]["suitcase-xdi"]["config"])
                self._xdi_file_template = toml.loads(
                    doc["md"]["suitcase-xdi"]["config"], _dict=OrderedDict
                )
            elif "config-file-path" in doc["md"]["suitcase-xdi"]:
                self._xdi_file_template = toml.load(
                    doc["md"]["suitcase-xdi"]["config-file-path"], _dict=OrderedDict
                )
            else:
                raise Exception(
                    "configuration must be specified as a file in md[suitcase-xdi][config-file-path]"
                    "or as a string in md[suitcase-xdi][config]"
                )
        else:
            # XDI file configuration was provided to __init__
            pass
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

        self._initialize_column_data_dict()
        # extract header information from the start document
        self._update_header_lines_from_doc(
            doc_name="start", doc=doc, header_line_buffer=self._header_line_buffer
        )

        # Fill in the file_prefix with the contents of the RunStart document.
        # As in, '{uid}' -> 'c1790369-e4b2-46c7-a294-7abfa239691a'
        # or 'my-data-from-{plan-name}' -> 'my-data-from-scan'
        self._templated_file_prefix = self._file_prefix.format(**doc)

        self.columns = tuple([v for k, v in self._xdi_file_template["columns"].items()])
        if len(self.columns) == 0:
            raise ValueError("found no Columns")

        self.export_data_keys = tuple({c["data_key"] for c in self.columns})

    def descriptor(self, doc):
        """
        It is possible to see more than one descriptor. Keep a list of all descriptors with data
        to be exported.

        Parameters
        ----------
        doc : dict
            an event-descriptor document
        """

        self._uid_to_descriptor[doc["uid"]] = doc
        print(f"got a descriptor for stream {doc['name']} with uid {doc['uid']}")
        self._update_header_lines_from_doc(
            doc_name="descriptor", doc=doc, header_line_buffer=self._header_line_buffer
        )

    def event_page(self, doc):
        """
        Maybe write a file. All data required for a file might be available.

        Parameters
        ----------
        doc : dict
            an event-page document
        """

        # get the stream name for this document
        stream_name = self._uid_to_descriptor[doc["descriptor"]]["name"]
        # print(
        #    f"event-page from stream {stream_name} with descriptor uid {doc['descriptor']}"
        # )

        # assumption: the primary stream event page arrives
        # between corresponding row_end stream event pages
        if stream_name == "row_ends":
            if len(self._row_end_docs) == 0:
                self._row_end_docs.append(doc)
                self._event_page_begin_row(doc)
            elif len(self._row_end_docs) == 1:
                self._event_page_end_row(doc)
                self._row_end_docs.clear()
            else:
                # something is wrong
                raise Exception(
                    f"self._row_end_docs should not have length {len(self._row_end_docs)}"
                )
        elif stream_name == "energy_bins":
            self._event_page_energy_bins(doc)
        elif stream_name == "primary":
            self._scan_number = doc["seq_num"]
            self._event_page_primary(doc)
        else:
            pass
            # self._event_page_other(doc)

    def _event_page_begin_row(self, doc):
        """
        Get the start time from this document.
        Parameters
        ----------
        """
        print("begin_row")
        self._event_page_header_line_buffer = {
            k: v for k, v in self._header_line_buffer.items() if v is None
        }

        if "Scan.start_time" in self._xdi_file_template["optional_headers"]:
            self._event_page_header_line_buffer[
                "Scan.start_time"
            ] = datetime.fromtimestamp(doc["time"]).isoformat()

    def _event_page_end_row(self, doc):
        """
        Get the end time from this document and write the file.
        Parameters
        ----------
        """
        if "Scan.end_time" in self._xdi_file_template["optional_headers"]:
            self._event_page_header_line_buffer[
                "Scan_time.end"
            ] = datetime.fromtimestamp(doc["time"]).isoformat()

    def _event_page_energy_bins(self, doc):
        self._update_data_columns_from_doc(doc=doc)

    def _event_page_primary(self, doc):
        """
        Parameters
        ----------

        """
        self._update_data_columns_from_doc(doc=doc)

        filename = self._templated_file_prefix + str(self._scan_number) + ".xdi"
        with self._manager.open("stream_data", filename, "xt") as xdi_file:
            # combine header line buffers maintaining header line order
            combined_header_line_buffer = dict(self._header_line_buffer)
            for k, v in self._event_page_header_line_buffer.items():
                if combined_header_line_buffer[k] is None:
                    combined_header_line_buffer[k] = v

            self._write_header(
                output_file=xdi_file, header_line_buffer=combined_header_line_buffer
            )
            # self._column_data_values looks like
            # [[...], [...], [...]]
            pprint(self._column_data)
            for row_data in zip_longest(*self._column_data.values(), fillvalue="NA"):
                xdi_file.write("\t".join((str(d) for d in row_data)))
                xdi_file.write("\n")

        # don't use this information again
        # self._column_data.clear()
        # self._initialize_column_data_dict()
        # self._event_page_header_line_buffer.clear()

    def _update_data_columns_from_doc(self, doc):
        # keep a dict of columns of data like:
        #  {
        #    "energy" : array...
        #    "I0"     : None
        #    "If"     : array...
        #  }
        for column in self.columns:
            data_key = column["data_key"]
            # expect to find an array for the data_key
            if data_key in doc["data"]:
                # if self._column_data[data_key] is None and data_key in doc["data"]:
                print(f"getting {data_key} from doc {doc['descriptor']}")
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
                        self._column_data[data_key] = (event_data,)

    def stop(self, doc):
        pass

    def _initialize_column_data_dict(self):
        print("_initialize_column_data_dict")
        # self._column_data = dict()
        for xdi_key, xdi_value in self._xdi_file_template["columns"].items():
            self._column_data[xdi_value["data_key"]] = None

    def _update_header_lines_from_doc(self, doc_name, doc, header_line_buffer):
        """
        Initialize or update the specified header_line_buffer using information
        in the specified document.

        The first time through header_line_buffer is empty. At this point
        all the XDI header keys from self._xdi_file_template will be inserted
        with value None into header_line_buffer.

        Subsequent calls will insert values (full XDI header rows) only for
        header line buffer entries with value None. Entries with a not-None
        header row value (a string) will not be updated.

        {}

        becomes

        {
          "XDI": None,
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

            Parameters
            ----------
            doc_name: str

            doc:

            header_line_buffer: dict[str, str]
                dict of (XDI header field, XDI header row) such as
                    ("Element.symbol", None)
                or
                    ("Element.symbol", "# Element.symbol = K")
        """

        def _get_empty_header_lines(config_section_name):
            """
            Return entries from self._xdi_file_template to be used to generate header rows
            using information from a document. Each (key, value) pair returned meets
            two criteria:
              1. 'key' is in both self._xdi_file_template and header_line_buffer
              2. header_line_buffer[key] is None (meaning this header row has not been generated yet)

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
                if _header_label not in header_line_buffer:
                    header_line_buffer[_header_label] = None
                else:
                    # _header_label is already a key in header_line_buffer
                    pass
                # if the corresponding header line has not been generated
                # yield the string template to generate that line
                if header_line_buffer[_header_label] is None:
                    yield _header_label, self._xdi_file_template[config_section_name][
                        _header_label
                    ]
                else:
                    # a header line string has already been generated for _header_label
                    pass

        for header_label, version in _get_empty_header_lines("versions"):
            header_line_buffer[header_label] = version

        for header_label, column_line_template in _get_empty_header_lines("columns"):
            header_value = column_line_template["column_label"].format(**doc)
            if "units" in column_line_template:
                header_line_buffer[header_label] = (
                    f"# {header_label} = {header_value} "
                    + column_line_template["units"]
                )
            else:
                header_line_buffer[header_label] = f"# {header_label} = {header_value}"

        for header_label, column_line_template in _get_empty_header_lines(
            "required_headers"
        ):
            print(doc_name)
            print(header_label)
            print(column_line_template)
            doc_name_unconstrained = "doc_name" not in column_line_template
            doc_name_constraint_satisfied = (
                "doc_name" in column_line_template
                and doc_name == column_line_template["doc_name"]
            )
            if doc_name_unconstrained or doc_name_constraint_satisfied:
                header_value = column_line_template["data"].format(**doc)
                header_line_buffer[header_label] = f"# {header_label} = {header_value}"

        for header_label, column_line_template in _get_empty_header_lines(
            "optional_headers"
        ):
            header_value = column_line_template["data"].format(**doc)
            header_line_buffer[header_label] = f"{header_label} = {header_value}"

    def _write_header(self, header_line_buffer, output_file):
        """Write all header information, "None" for missing information.

        """
        for header_field, header_value in header_line_buffer.items():
            output_file.write(header_value)
            output_file.write("\n")

        output_file.write("#----\n")
        output_file.write(
            "# {}\n".format("\t".join([c["column_label"] for c in self.columns]))
        )
