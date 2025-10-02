from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'actions/project_index/stream_create_document.jinja.html'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    pass
    yield '<turbo-stream action="update" target="frame_project_tree">\n  <template>\n    '
    template = environment.get_template('screens/project_index/project_tree.jinja', 'actions/project_index/stream_create_document.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    yield '\n  </template>\n</turbo-stream>\n\n<turbo-stream action="update" target="modal">\n  <template></template>\n</turbo-stream>'

blocks = {}
debug_info = '3=12'