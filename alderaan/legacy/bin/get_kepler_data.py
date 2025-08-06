import argparse
import os
import stat


def lookup_epochs(quarter, cadence):
    LONG_QUARTER_PREFIXES = {
        "0": ["2009131105131"],
        "1": ["2009166043257"],
        "2": ["2009259160929"],
        "3": ["2009350155506"],
        "4": ["2010078095331", "2010009091648"],
        "5": ["2010174085026"],
        "6": ["2010265121752"],
        "7": ["2010355172524"],
        "8": ["2011073133259"],
        "9": ["2011177032512"],
        "10": ["2011271113734"],
        "11": ["2012004120508"],
        "12": ["2012088054726"],
        "13": ["2012179063303"],
        "14": ["2012277125453"],
        "15": ["2013011073258"],
        "16": ["2013098041711"],
        "17": ["2013131215648"],
    }

    SHORT_QUARTER_PREFIXES = {
        "0": ["2009131110544"],
        "1": ["2009166044711"],
        "2": ["2009201121230", "2009231120729", "2009259162342"],
        "3": ["2009291181958", "2009322144938", "2009350160919"],
        "4": ["2010009094841", "2010019161129", "2010049094358", "2010078100744"],
        "5": ["2010111051353", "2010140023957", "2010174090439"],
        "6": ["2010203174610", "2010234115140", "2010265121752"],
        "7": ["2010296114515", "2010326094124", "2010355172524"],
        "8": ["2011024051157", "2011053090032", "2011073133259"],
        "9": ["2011116030358", "2011145075126", "2011177032512"],
        "10": ["2011208035123", "2011240104155", "2011271113734"],
        "11": ["2011303113607", "2011334093404", "2012004120508"],
        "12": ["2012032013838", "2012060035710", "2012088054726"],
        "13": ["2012121044856", "2012151031540", "2012179063303"],
        "14": ["2012211050319", "2012242122129", "2012277125453"],
        "15": ["2012310112549", "2012341132017", "2013011073258"],
        "16": ["2013017113907", "2013065031647", "2013098041711"],
        "17": ["2013121191144", "2013131215648"],
    }
    if cadence == "long":
        if quarter in LONG_QUARTER_PREFIXES:
            return LONG_QUARTER_PREFIXES[quarter]
        else:
            raise ValueError(
                "*** ERROR in lookup_epochs: Quarter must be between 0 and 17."
            )
    elif cadence == "short":
        if quarter in SHORT_QUARTER_PREFIXES:
            return SHORT_QUARTER_PREFIXES[quarter]
        else:
            raise ValueError(
                "*** ERROR in lookup_epochs: Quarter must be between 0 and 17."
            )
    else:
        raise ValueError(
            "*** ERROR in lookup_epochs: Cadence must be 'long' or 'short'."
        )


def print_cmd(ofile, url, cadence_str, kepid, epochs, cmdtype):
    # Print a data download command to an opened file (LUN) given a Kepler ID and array of epochs.
    for epoch in epochs:
        this_filename = "kplr" + kepid + "-" + epoch + cadence_str
        if cmdtype == "curl":
            cmd_str_to_add = " -f -R -o " + this_filename + " "
        else:
            cmd_str_to_add = " "
        this_cmd = (
            cmdtype
            + cmd_str_to_add
            + url
            + kepid[0:4]
            + "/"
            + kepid
            + "/"
            + this_filename
        )
        ofile.write(this_cmd + "\n")


def get_kepler(idict):
    # Extract command-line arguments from the input dict.
    kepids = idict["kepids"]
    cadence = idict["cadence"]
    data_type = idict["data_type"]
    ofile = idict["ofile"]
    cmdtype = idict["cmdtype"]
    if "epochs" in idict:
        epochs = idict["epochs"]
    else:
        epochs = None
    if "quarters" in idict:
        quarters = idict["quarters"]
    else:
        quarters = None

    # This is the base URL to use when requesting Kepler files.
    lc_base_url = "http://archive.stsci.edu/missions/kepler/lightcurves/"
    tp_base_url = "http://archive.stsci.edu/missions/kepler/target_pixel_files/"
    if data_type == "lightcurve":
        base_url = lc_base_url
    else:
        base_url = tp_base_url

    # Define the file extensions to use.
    if data_type == "lightcurve":
        if cadence == "long":
            cadence_str = "_llc.fits"
        else:
            cadence_str = "_slc.fits"
    else:
        if cadence == "long":
            cadence_str = "_lpd-targ.fits.gz"
        else:
            cadence_str = "_spd-targ.fits.gz"

    # Use all available quarters if neither 'epochs' or 'quarters' are specified.
    if epochs is None and quarters is None:
        quarters = [str(x) for x in range(18)]

    # If 'kepids' is a scalar string put it in a list.  Same with 'epochs' and 'quarters'
    if isinstance(kepids, str):
        kepids = [kepids]
    if isinstance(epochs, str):
        epochs = [epochs]
    if isinstance(quarters, str):
        quarters = [quarters]

    # Make sure Kepler IDs are zero-padded.
    kepids = ["{0:09d}".format(int(x)) for x in kepids]

    # Write to script file.
    with open(ofile, "w") as output_file:
        output_file.write("#!/bin/sh\n")
        for kepid in kepids:
            if quarters is not None:
                for quarter in quarters:
                    epoch_strings = lookup_epochs(quarter, cadence)
                    print_cmd(
                        output_file,
                        base_url,
                        cadence_str,
                        kepid,
                        epoch_strings,
                        cmdtype,
                    )
            else:
                print_cmd(output_file, base_url, cadence_str, kepid, epochs, cmdtype)
        # Print a warning message to let users know there will sometimes be Error 404's.
        output_file.write("echo\n")
        output_file.write(
            'echo Script completed.  Note: Some \\"Error 404\\" messages are expected,'
            " depending on your parameters.  It is always recommended to confirm the"
            " expected number of files are successfully downloaded.\n"
        )

    # Make sure file is executable.
    os.chmod(ofile, 0o744)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to generate download commands given one or"
        " more Kepler IDs.",
        epilog="Example: python get_kepler.py"
        " 7730747 7748238 -c short -t lightcurve -q 7 8",
    )
    parser.add_argument(
        "kepids",
        action="store",
        nargs="+",
        help="One or more Kepler IDs to" " retrieve data from.",
    )
    parser.add_argument(
        "-c",
        action="store",
        type=str.lower,
        dest="cadence",
        default="long",
        choices=["short", "long"],
        help="Specify the type of cadence.  Default=" "'%(default)s'.",
    )
    parser.add_argument(
        "-t",
        action="store",
        type=str.lower,
        dest="data_type",
        default="lightcurve",
        choices=["lightcurve", "target_pixel_file"],
        help="What type" " of data product to retrieve.  Default = '%(default)s'.",
    )
    parser.add_argument(
        "-o",
        action="store",
        dest="ofile",
        default="get_kepler_data.sh",
        help="Full"
        " path and file name of the output file to create.  Any existing file with"
        " the same name will be overwritten.  Default = '%(default)s'.",
    )
    parser.add_argument(
        "--cmdtype",
        action="store",
        type=str.lower,
        dest="cmdtype",
        default="curl",
        choices=["curl", "wget"],
        help="What download tool should be used?  Default" " = '%(default)s'.",
    )
    epoch_quarter_group = parser.add_mutually_exclusive_group()
    epoch_quarter_group.add_argument(
        "-e",
        action="store",
        type=str,
        dest="epochs",
        nargs="*",
        help="One or more epochs to retrieve with each Kepler ID.",
    )
    epoch_quarter_group.add_argument(
        "-q",
        action="store",
        type=str,
        dest="quarters",
        nargs="*",
        help="One or more quarters to retrieve with each Kepler ID.",
    )
    args = vars(parser.parse_args())
    get_kepler(args)
