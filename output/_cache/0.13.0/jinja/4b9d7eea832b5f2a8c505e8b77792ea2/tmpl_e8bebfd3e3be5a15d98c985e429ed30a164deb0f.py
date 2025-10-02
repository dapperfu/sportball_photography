from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'components/node/readonly.jinja'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_sdoc_entity = resolve('sdoc_entity')
    l_0_node_type_string = l_0__user_requirement_style = l_0__has_multiline_fields = l_0__narrative_has_no_multiline_fields = missing
    try:
        t_1 = environment.tests['none']
    except KeyError:
        @internalcode
        def t_1(*unused):
            raise TemplateRuntimeError("No test named 'none' found.")
    pass
    yield '\n\n'
    l_0_node_type_string = context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'get_node_type_string'))
    context.vars['node_type_string'] = l_0_node_type_string
    context.exported_vars.add('node_type_string')
    yield '\n\n\n'
    l_0__user_requirement_style = context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'get_requirement_style_mode'))
    context.vars['_user_requirement_style'] = l_0__user_requirement_style
    yield '\n'
    l_0__has_multiline_fields = context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'has_multiline_fields'))
    context.vars['_has_multiline_fields'] = l_0__has_multiline_fields
    yield '\n'
    l_0__narrative_has_no_multiline_fields = ((not (undefined(name='_has_multiline_fields') if l_0__has_multiline_fields is missing else l_0__has_multiline_fields)) and ((undefined(name='_user_requirement_style') if l_0__user_requirement_style is missing else l_0__user_requirement_style) == 'narrative'))
    context.vars['_narrative_has_no_multiline_fields'] = l_0__narrative_has_no_multiline_fields
    yield '\n\n  <sdoc-node\n    node-style="readonly"\n    node-role="'
    yield escape(context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'get_type_string')))
    yield '"'
    if (not t_1((undefined(name='node_type_string') if l_0_node_type_string is missing else l_0_node_type_string))):
        pass
        yield '\n      show-node-type-name="'
        yield escape((undefined(name='node_type_string') if l_0_node_type_string is missing else l_0_node_type_string))
        yield '"'
    yield '\n    '
    if environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'is_requirement'):
        pass
        yield '\n      node-view="'
        yield escape((undefined(name='_user_requirement_style') if l_0__user_requirement_style is missing else l_0__user_requirement_style))
        yield '"'
    if (undefined(name='_narrative_has_no_multiline_fields') if l_0__narrative_has_no_multiline_fields is missing else l_0__narrative_has_no_multiline_fields):
        pass
        yield '\n      class="html2pdf4doc-no-break"'
    yield '\n    data-testid="node-'
    yield escape(context.call(environment.getattr((undefined(name='sdoc_entity') if l_0_sdoc_entity is missing else l_0_sdoc_entity), 'get_type_string')))
    yield '"\n  >\n\n    '
    template = environment.get_template('components/anchor/index.jinja', 'components/node/readonly.jinja')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'_has_multiline_fields': l_0__has_multiline_fields, '_narrative_has_no_multiline_fields': l_0__narrative_has_no_multiline_fields, '_user_requirement_style': l_0__user_requirement_style, 'node_type_string': l_0_node_type_string}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    yield '\n    \n    '
    yield from context.blocks['sdoc_entity'][0](context)
    yield '\n\n  </sdoc-node>'

def block_sdoc_entity(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    _block_vars = {}
    pass
    yield '\n    \n    '

blocks = {'sdoc_entity': block_sdoc_entity}
debug_info = '3=20&17=24&18=27&19=30&23=33&24=35&25=38&27=41&28=44&30=46&33=50&36=52&38=59'