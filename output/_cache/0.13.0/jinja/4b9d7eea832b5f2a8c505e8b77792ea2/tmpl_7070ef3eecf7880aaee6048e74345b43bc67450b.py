from jinja2.runtime import LoopContext, Macro, Markup, Namespace, TemplateNotFound, TemplateReference, TemplateRuntimeError, Undefined, escape, identity, internalcode, markup_join, missing, str_join
name = 'screens/document/document/frame_document_config_edit.jinja.html'

def root(context, missing=missing):
    resolve = context.resolve_or_missing
    undefined = environment.undefined
    concat = environment.concat
    cond_expr_undefined = Undefined
    if 0: yield None
    l_0_document = resolve('document')
    l_0_form_object = resolve('form_object')
    l_0_namespace = resolve('namespace')
    l_0_custom_metadata_row_context = missing
    pass
    yield '<turbo-frame id="article-'
    yield escape(environment.getattr((undefined(name='document') if l_0_document is missing else l_0_document), 'reserved_mid'))
    yield '">\n<sdoc-form>\n\n  <form\n    method="POST"\n    data-turbo="true"\n    action="/actions/document/save_config"\n    data-controller="scroll_into_view tabs"\n  >\n    <input type="hidden" id="document_mid" name="document_mid" value="'
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_mid'))
    yield '"/>\n\n    \n    <sdoc-tab-content id="Document" active>\n      <sdoc-form-grid>'
    l_1_field_type = 'singleline'
    l_1_field_name = 'TITLE'
    l_1_field_input_name = 'document[TITLE]'
    l_1_field_value = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_title')
    l_1_errors = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), 'TITLE')
    pass
    template = environment.get_template('components/form/field/text/index.jinja', 'screens/document/document/frame_document_config_edit.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'errors': l_1_errors, 'field_input_name': l_1_field_input_name, 'field_name': l_1_field_name, 'field_type': l_1_field_type, 'field_value': l_1_field_value, 'custom_metadata_row_context': l_0_custom_metadata_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    l_1_field_type = l_1_field_name = l_1_field_input_name = l_1_field_value = l_1_errors = missing
    l_1_field_type = 'singleline'
    l_1_field_name = 'UID'
    l_1_field_input_name = 'document[UID]'
    l_1_field_value = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_uid')
    l_1_errors = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), 'UID')
    pass
    template = environment.get_template('components/form/field/text/index.jinja', 'screens/document/document/frame_document_config_edit.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'errors': l_1_errors, 'field_input_name': l_1_field_input_name, 'field_name': l_1_field_name, 'field_type': l_1_field_type, 'field_value': l_1_field_value, 'custom_metadata_row_context': l_0_custom_metadata_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    l_1_field_type = l_1_field_name = l_1_field_input_name = l_1_field_value = l_1_errors = missing
    l_1_field_type = 'singleline'
    l_1_field_name = 'VERSION'
    l_1_field_input_name = 'document[VERSION]'
    l_1_field_value = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_version')
    l_1_errors = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), 'VERSION')
    pass
    template = environment.get_template('components/form/field/text/index.jinja', 'screens/document/document/frame_document_config_edit.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'errors': l_1_errors, 'field_input_name': l_1_field_input_name, 'field_name': l_1_field_name, 'field_type': l_1_field_type, 'field_value': l_1_field_value, 'custom_metadata_row_context': l_0_custom_metadata_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    l_1_field_type = l_1_field_name = l_1_field_input_name = l_1_field_value = l_1_errors = missing
    l_1_field_type = 'singleline'
    l_1_field_name = 'CLASSIFICATION'
    l_1_field_input_name = 'document[CLASSIFICATION]'
    l_1_field_value = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_classification')
    l_1_errors = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), 'CLASSIFICATION')
    pass
    template = environment.get_template('components/form/field/text/index.jinja', 'screens/document/document/frame_document_config_edit.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'errors': l_1_errors, 'field_input_name': l_1_field_input_name, 'field_name': l_1_field_name, 'field_type': l_1_field_type, 'field_value': l_1_field_value, 'custom_metadata_row_context': l_0_custom_metadata_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    l_1_field_type = l_1_field_name = l_1_field_input_name = l_1_field_value = l_1_errors = missing
    l_1_field_type = 'singleline'
    l_1_field_name = 'PREFIX'
    l_1_field_input_name = 'document[PREFIX]'
    l_1_field_value = environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_requirement_prefix')
    l_1_errors = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), 'PREFIX')
    pass
    template = environment.get_template('components/form/field/text/index.jinja', 'screens/document/document/frame_document_config_edit.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'errors': l_1_errors, 'field_input_name': l_1_field_input_name, 'field_name': l_1_field_name, 'field_type': l_1_field_type, 'field_value': l_1_field_value, 'custom_metadata_row_context': l_0_custom_metadata_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    l_1_field_type = l_1_field_name = l_1_field_input_name = l_1_field_value = l_1_errors = missing
    yield '</sdoc-form-grid>\n\n    </sdoc-tab-content>\n\n    \n    <sdoc-tab-content id="Metadata">\n      <sdoc-form-grid>\n\n        <div style="display: contents;" id="document_'
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_mid'))
    yield '__new_metadata">\n          '
    l_0_custom_metadata_row_context = context.call((undefined(name='namespace') if l_0_namespace is missing else l_0_namespace))
    context.vars['custom_metadata_row_context'] = l_0_custom_metadata_row_context
    context.exported_vars.add('custom_metadata_row_context')
    yield '\n          '
    for l_1_metadata_field in environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'custom_metadata_fields'):
        _loop_vars = {}
        pass
        yield '\n            '
        if not isinstance(l_0_custom_metadata_row_context, Namespace):
            raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
        l_0_custom_metadata_row_context['field'] = l_1_metadata_field
        yield '\n            '
        if not isinstance(l_0_custom_metadata_row_context, Namespace):
            raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
        l_0_custom_metadata_row_context['form_object'] = (undefined(name='form_object') if l_0_form_object is missing else l_0_form_object)
        yield '\n            '
        if not isinstance(l_0_custom_metadata_row_context, Namespace):
            raise TemplateRuntimeError("cannot assign attribute on non-namespace object")
        l_0_custom_metadata_row_context['errors'] = context.call(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'get_errors'), markup_join(('METADATA[', environment.getattr(l_1_metadata_field, 'field_mid'), ']', )), _loop_vars=_loop_vars)
        yield '\n            '
        template = environment.get_template('components/form/row/row_with_metadata.jinja', 'screens/document/document/frame_document_config_edit.jinja.html')
        gen = template.root_render_func(template.new_context(context.get_all(), True, {'metadata_field': l_1_metadata_field, 'custom_metadata_row_context': l_0_custom_metadata_row_context}))
        try:
            for event in gen:
                yield event
        finally: gen.close()
    l_1_metadata_field = missing
    yield '</div>\n      </sdoc-form-grid>\n\n      <sdoc-form-footer>\n\n      <a\n        class="action_button"\n        href="/actions/document/new_metadata?document_mid='
    yield escape(environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_mid'))
    yield '"\n        data-turbo-action="replace"\n        data-turbo="true"\n        data-action-type="add_field"\n        data-testid="form-action-add-metadata-field"\n      >'
    template = environment.get_template('_res/svg_ico16_add.jinja.html', 'screens/document/document/frame_document_config_edit.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'custom_metadata_row_context': l_0_custom_metadata_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    yield ' Add new metadata value</a>\n      </sdoc-form-footer>\n\n    </sdoc-tab-content>   \n\n    <sdoc-form-footer>\n      '
    template = environment.get_template('components/button/submit.jinja', 'screens/document/document/frame_document_config_edit.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'custom_metadata_row_context': l_0_custom_metadata_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    l_1_cancel_href = markup_join(('/actions/document/cancel_edit_config?document_mid=', environment.getattr((undefined(name='form_object') if l_0_form_object is missing else l_0_form_object), 'document_mid'), ))
    pass
    template = environment.get_template('components/button/cancel.jinja', 'screens/document/document/frame_document_config_edit.jinja.html')
    gen = template.root_render_func(template.new_context(context.get_all(), True, {'cancel_href': l_1_cancel_href, 'custom_metadata_row_context': l_0_custom_metadata_row_context}))
    try:
        for event in gen:
            yield event
    finally: gen.close()
    l_1_cancel_href = missing
    yield '</sdoc-form-footer>\n\n  </form>\n\n</sdoc-form>\n</turbo-frame>'

blocks = {}
debug_info = '1=16&10=18&23=26&33=39&43=52&53=65&63=78&74=86&75=88&76=92&77=98&78=102&79=106&80=108&89=116&94=118&100=125&102=133'