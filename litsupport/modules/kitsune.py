# This module should always be added *after* timeit

from litsupport import shellcommand
from litsupport import testplan

def _mutateCommandLine(context, commandline):
    cmd = shellcommand.parse(commandline)

    # This is a pretty lousy way of doing things, but if there is a nicer way,
    # I cannot seem to find it. The timeit module automatically adds a number of
    # arguments to limit CPU time, resources used etc. Some of our tests easily
    # exceed these. In particular, the OpenCilk backend uses all available cores
    # with the result that the limit on CPU seconds is hit very quickly. Since
    # the timeit module is broadly useful, we don't want to disable it
    # altogether. We should probably find a way to make the timeouts and
    # things configurable instead, but for now, we will just go in and remove
    # any limits that we don't like.
    to_remove = []
    for i, arg in enumerate(cmd.arguments):
        if arg == '--limit-cpu':
            to_remove.append(i)
            to_remove.append(i + 1)

    # Remove everything in reverse. If we remove in the order in which these
    # were inserted, the indices would be invalid. We really should use
    # collections.deque here, but let's keep the hack "hacky".
    to_remove.reverse()
    for i in to_remove:
        del cmd.arguments[i]

    return cmd.toCommandline()


def _mutateScript(context, script):
    return testplan.mutateScript(context, script, _mutateCommandLine)


def mutatePlan(context, plan):
    if len(plan.runscript) == 0:
        return
    context.timefiles = []
    plan.runscript = _mutateScript(context, plan.runscript)
