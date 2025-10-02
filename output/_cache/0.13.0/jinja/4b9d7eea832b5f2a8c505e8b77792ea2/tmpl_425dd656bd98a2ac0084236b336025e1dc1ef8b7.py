from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'actions/document/delete_requirement/stream_confirm_delete_requirement.jinja'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_default = resolve('default')
    l_0_requirement_mid = resolve('requirement_mid')
    l_0_context_document_mid = resolve('context_document_mid')
    pass
    yield '<turbo-stream action="update" target="confirm">\n  <template>\n  '
    l_1_confirm_title = (undefined(name='default') if l_0_default is missing else l_0_default)
    l_1_confirm_message = 'Deleting a requirement is an unrecoverable action.'
    l_1_confirm_name = 'Delete requirement'
    l_1_confirm_href = markup_join(('/actions/document/delete_requirement?node_id=', (undefined(name='requirement_mid') if l_0_requirement_mid is missing else l_0_requirement_mid), '&context_document_mid=', (undefined(name='context_document_mid') if l_0_context_document_mid is missing else l_0_context_document_mid), '&confirmed=1', ))
    pass
    yield '\n    '
    template = environment.get_template('components/confirm/index.jinja', 'actions/document/delete_requirement/stream_confirm_delete_requirement.jinja')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'confirm_href': l_1_confirm_href, 'confirm_message': l_1_confirm_message, 'confirm_name': l_1_confirm_name, 'confirm_title': l_1_confirm_title}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    yield '\n  '
    l_1_confirm_title = l_1_confirm_message = l_1_confirm_name = l_1_confirm_href = missing
    yield '\n  </template>\n</turbo-stream>'

blocks = {}
debug_info = '9=21'