# -*- coding: utf-8 -*-
def main():  # nocover
    import wbia_blend

    print('Looks like the imports worked')
    print('wbia_blend = {!r}'.format(wbia_blend))
    print('wbia_blend.__file__ = {!r}'.format(wbia_blend.__file__))
    print('wbia_blend__version__ = {!r}'.format(wbia_blend.__version__))


if __name__ == '__main__':
    """
    CommandLine:
       python -m wbia_blend
    """
    main()
