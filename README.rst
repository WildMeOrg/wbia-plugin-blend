==========================
Wildbook IA - wbia_pie_v2
==========================

**The repository is forked and further updated/maintained with** `WildMe <https://github.com/WildMeOrg/wbia-plugin-pie-v2>`_

Pose Invariant Embedding Re-identification Plug-in - Part of the WildMe / Wildbook IA Project.

A plugin for re-identification of wildlife individuals based on unique natural body
markings. Updated implementation with PyTorch (`first version here <https://github.com/WildMeOrg/wbia-plugin-pie>`_).

Installation
------------

.. code:: bash

    ./run_developer_setup.sh

REST API
--------

With the plugin installed, register the module name with the `WBIAControl.py` file
in the wbia repository located at `wbia/wbia/control/WBIAControl.py`.  Register
the module by adding the string (for example, `wbia_plugin_identification_example`) to the
list `AUTOLOAD_PLUGIN_MODNAMES`.

Then, load the web-based WBIA IA service and open the URL that is registered with
the `@register_api decorator`.

.. code:: bash

    cd ~/code/wbia/
    python dev.py --web

Navigate in a browser to http://127.0.0.1:5000/api/plugin/example/helloworld/ where
this returns a formatted JSON response, including the serialized returned value
from the `wbia_plugin_identification_example_hello_world()` function

.. code:: text

    {"status": {"cache": -1, "message": "", "code": 200, "success": true}, "response": "[wbia_plugin_identification_example] hello world with WBIA controller <WBIAController(testdb1) at 0x11e776e90>"}

Python API
----------

.. code:: bash

    python
    >>> import wbia_pie_v2
    >>> from wbia_pie_v2._plugin import DEMOS, CONFIGS, MODELS
    >>> species = 'whale_shark'
    >>> test_ibs = wbia_pie_v2._plugin.wbia_pie_v2_test_ibs(DEMOS[species], species, 'test2021')
    >>> aid_list = test_ibs.get_valid_aids(species=species)
    >>> rank1 = test_ibs.evaluate_distmat(aid_list, CONFIGS[species], use_depc=False)
    >>> expected_rank1 = 0.81366
    >>> assert abs(rank1 - expected_rank1) < 1e-2

The function from the plugin is automatically added as a method to the ibs object
as `ibs.pie_embedding()`, which is registered using the
`@register_ibs_method decorator`.

Code Style and Development Guidelines
-------------------------------------

Contributing
~~~~~~~~~~~~

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any commit you make. (See also `pre-commit.com <https://pre-commit.com/>`_)

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code:: bash

    git add .
    pre-commit run

    # or

    pre-commit run --all-files

Brunette
~~~~~~~~

Our code base has been formatted by Brunette, which is a fork and more configurable version of Black (https://black.readthedocs.io/en/stable/).

Flake8
~~~~~~

Try to conform to PEP8.  You should set up your preferred editor to use flake8 as its Python linter, but pre-commit will ensure compliance before a git commit is completed.

To run flake8 from the command line use:

.. code:: bash

    flake8


This will use the flake8 configuration within ``setup.cfg``,
which ignores several errors and stylistic considerations.
See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.

PyTest
~~~~~~

Our code uses Google-style documentation tests (doctests) that uses pytest and xdoctest to enable full support.  To run the tests from the command line use:

.. code:: bash

    pytest

To run doctests with `+REQUIRES(--web-tests)` do:

.. code:: bash

    pytest --web-tests


Results and Examples
---------------------

Quantitative and qualitative results are presented `here </wbia_pie_v2>`_


Implementation details
----------------------
Dependencies
~~~~~~~~~~~~~
* Python >= 3.7
* PyTorch >= 1.5
* Torchvision >= 0.8

Source Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Key annotations required:

* bounding box containing a pattern of interest
* unique name of an animal individual

Training
~~~~~~~~~~~~

Run the training script:

.. code:: bash

    cd wbia_pie_v2
    python train.py --cfg <path_to_config_file> <additional_optional_params>

Configuration files are listed in ``wbia_pie_v2/configs`` folder. For example, the following line trains the model with parameters specified in the config file:

.. code:: bash

    python train.py --cfg configs/01_whaleshark_cropped_resnet50.yaml


To override a parameter in config, add this parameter as a command line argument:

.. code:: bash

    python train.py --cfg configs/01_whaleshark_cropped_resnet50.yaml train.batch_size 48

To evaluate a model on the test subset, set the parameter ``test.evaluate True`` and
parameter ``test.visrank True`` to visualize results.
Provide a path to the model saved during training.
For example:

.. code:: bash

    python train.py --cfg configs/01_whaleshark_cropped_resnet50.yaml test.evaluate True model.load_weights <path_to_trained_model>


Acknowledgement
---------------

The code is adapted from `TorchReid <https://github.com/KaiyangZhou/deep-person-reid>`_ library for person re-identification.
