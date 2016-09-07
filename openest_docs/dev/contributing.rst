.. _contributing:

Contributor's Guide
===================


This document lays out guidelines and advice for contributing to this project.
If you're thinking of contributing, please start by reading this document and
getting a feel for how contributing to this project works. If you have any
questions, feel free to reach out to either `James Rising`_, `Mike Delgado`_, or `Justin Simcock`_.

.. _James Rising: jarising@gmail.com
.. _Mike Delgado: mdelgado@rhg.com
.. _Justin Simcock: jsimcock@rhg.com



When contributing code, you'll want to follow this checklist:

1. Fork the repository on GitHub.
2. Run the tests to confirm they all pass on your system. If they don't, you'll
   need to investigate why they fail. If you're unable to diagnose this
   yourself, raise it as a bug report by following the guidelines in this
   document: :ref:`bug-reports`. 
3. Write tests that demonstrate your bug or feature. Ensure that they fail.
4. Make your change.
5. Run the entire test suite again, confirming that all tests pass *including
   the ones you just added*.
6. Send a GitHub Pull Request to the main repository's ``master`` branch.
   GitHub Pull Requests are the expected method of code collaboration on this
   project.



Documentation Links
-------------------

The documentation files live in the ``openest_docs/`` directory of the codebase. 
They're written in `reStructuredText`_, and use `Sphinx`_ to generate the full suite of
documentation.

When writing documentation, please do your best to follow the style of the
documentation files. An :doc:` example <documenation_example>` can be seen here.

.. _reStructuredText: http://docutils.sourceforge.net/rst.html
.. _Sphinx: http://sphinx-doc.org/index.html




.. _bug-reports:

Bug Reports
-----------

Bug reports are hugely important! Before you raise one, though, please check
through the `GitHub issues`_, **both open and closed**, to confirm that the bug
hasn't been reported before. Duplicate bug reports are a huge drain on the time
of other contributors, and should be avoided as much as possible.

.. _GitHub issues: https://github.com/jrising/open-estimate/issues

